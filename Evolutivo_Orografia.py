# -*- coding: utf-8 -*-
"""
@author: Sergio

1º -> Activamos el entorno de conda -> conda activate TFM_MUIT
2º -> Podemos ejecutar el código sin problemas
"""
import rasterio
from rasterio.transform import rowcol, from_origin
from pyproj import CRS, Transformer
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import os
import csv

class EvolutiveClass:
    def __init__(self, Num_Individuos=200, Num_Generaciones=10, Tam_Individuos=1, Num_Max = 10, Prob_Padres=0.5, Prob_Mutacion=0.02, Prob_Cruce=0.5):
        self.Num_Individuos = Num_Individuos    #Número de posibles soluciones
        self.Num_Generaciones = Num_Generaciones
        self.Tam_Individuos = Tam_Individuos    #Número de bases
        self.Num_Max = Num_Max  #Número de SD
        self.Prob_Padres = Prob_Padres
        self.Num_Padres = round(self.Num_Individuos * self.Prob_Padres)
        self.Prob_Mutacion = Prob_Mutacion
        self.Prob_Cruce = Prob_Cruce

    def ImprimirInformacion(self):
        print("Los parámetros del algoritmo genético son los siguientes:")
        print("Número de individuos: " + str(self.Num_Individuos))
        print("Número de generaciones: " + str(self.Num_Generaciones))
        print("Probabilidad de padres que sobreviven: " + str(self.Prob_Padres))
        print("Número de padres: " + str(self.Num_Padres))
        print("Probabilidad de mutación: " + str(self.Prob_Mutacion))
        print("Probabilidad de cruce: " + str(self.Prob_Cruce))
    
    def PoblacionInicial(self, Capacidades, Fil=None, Col=None, Num_Max=None):
        if Fil == None:
            Fil = self.Num_Individuos
        if Col == None:
            Col = self.Tam_Individuos
        if Num_Max == None:
            Num_Max = self.Num_Max
        Pob_Ini = np.random.randint(0,Num_Max, size=(Fil,Col))  #Son los índices de los SD asignados a cada base
        for i in range(Fil):    #Comprobamos todos los individuos y los reparamos si estuvieran mal
            if(self.Comprobacion_Individuo(Pob_Ini[i], Capacidades)):
                Pob_Ini[i] = self.Reparacion_Mayor_Menor(Pob_Ini[i], Capacidades)
        return Pob_Ini

    def Seleccion(self, poblacion_inicial, coste):
        index = np.argsort(coste)
        coste_ordenado = np.sort(coste)
        poblacion_actual = poblacion_inicial[index,:]   #La población tendrá más soluciones que la inicial debido al cruce
        poblacion_actual = poblacion_actual[0:self.Num_Individuos,:]    #Nos quedamos con los mejores individuos
        return poblacion_actual, coste_ordenado

    def Cruce (self, poblacion, capacidades, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        #Indice_Seleccionado = []
        Indices_Validos = list(np.arange(self.Num_Individuos))

        for i in range(self.Num_Individuos - self.Num_Padres): #Bucle para generar HIJOS
            Indice_Padres = random.sample(Indices_Validos, 2)
            #Indice_Padres = random.sample([j for j in Indices_Validos if j not in Indice_Seleccionado], 2)            # Se elige aleatoriamente el indice de los padres
            #Indice_Seleccionado.extend(Indice_Padres)   #Guardamos los índices elegidos para que no los vuelva a repetir en la siguiente iteración
            Padre1 = poblacion[Indice_Padres[0],:]                              # Se coge el padre 1
            Padre2 = poblacion[Indice_Padres[1],:]                              # Se coge el padre 2
            Hijo = np.copy(Padre1)                                              # El hijo va a ser una copia del padre 1
            vector = 1*(np.random.rand(self.Tam_Individuos) > self.Prob_Cruce)  # Se genera un vector para seleccionar los genes del padre 2
            Hijo[np.where(vector==1)[0]] = Padre2[np.where(vector==1)[0]]       # Los genes seleccionados del padre 2 pasan al hijo
            if np.random.rand() < self.Prob_Mutacion:                           # Se comprueba si el hijo va a mutar
                Hijo = self.Mutacion(Hijo, Num_Max)
            if(self.Comprobacion_Individuo(Hijo, capacidades)):                    # Se comprueba si hay que reparar el hijo
                 Hijo = self.Reparacion_Mayor_Menor(Hijo, capacidades)
            poblacion = np.insert(poblacion,self.Num_Padres+i,Hijo, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado
        return poblacion

    def Mutacion (self, individuo, Num_Max=None):                                
        aux1 = np.random.randint(0, individuo.shape[0])                         # Se genera número aleatorio para ver la posición que muta
        aux2 = np.random.randint(0,Num_Max)                                   # Se genera el número a modificar
        individuo[aux1] = aux2        
        return individuo
    def Comprobacion_Individuo (self, individuo, capacidades):
        suma_comprobar = list(np.zeros(self.Num_Max))
        for i in range(self.Num_Max):
            indices_bases = [j for j, value in enumerate(individuo) if value == i]  #Obtenemos los indices de las bases asociadas a un SD "i"
            comprobar_capacidades = capacidades[indices_bases]
            suma_comprobar[i] = sum(comprobar_capacidades)
            Caps_Comprobar = [ind_cap for ind_cap, j in enumerate(suma_comprobar) if j > 200]
            if len(Caps_Comprobar) > 0:
                return True
    def Reparacion_Aleatorio (self, individuo, capacidades): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
        capacidades_sd = list(np.zeros(self.Num_Max))  # Capacidades de los SD
        suma_capacidades = list(np.zeros(self.Num_Max))  # Suma de las capacidades de las bases
        for i in range(self.Num_Max):
            indices_bases_reparar = [j for j, value in enumerate(individuo) if value == i]
            capacidades_sd_i = capacidades[indices_bases_reparar]
            capacidades_sd[i] = capacidades_sd_i
            suma_capacidades[i] = sum(capacidades_sd[i])  # Almacenamos todas las sumas de las capacidades en un array
        Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad
        if len(Caps_SD_Superadas) > 0:
            while True:
                k = np.random.randint(0, numero_supply_depots)
                indices_bases_SD = [j for j, value in enumerate(individuo) if value == k]  # Obtenemos índices de las bases cuya suma de caps supera el umbral
                indices_resto_bases = [j for j, value in enumerate(individuo) if value != k]  # Obtenemos índices del resto de bases

                indice_base_1 = random.choice(indices_bases_SD)  # Elegimos una de las 5 bases del SD con mayor capacidad
                indice_base_aleatoria_2 = random.choice(indices_resto_bases)  # Elección aleatoria de la base del resto de bases

                individuo[indice_base_1], individuo[indice_base_aleatoria_2] = individuo[indice_base_aleatoria_2],individuo[indice_base_1]  # Intercambio posiciones de las bases
                for i in range(numero_supply_depots):  # Bucle para comprobar sumas de capacidades alrededor de un SD
                    F = np.array([t for t, x in enumerate(individuo) if x == i], dtype=int)
                    I = np.array(capacidades)
                    suma_capacidades[i] = sum(I[F])
                Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad
                if len(Caps_SD_Superadas) == 0:
                    break
        return individuo

    def Reparacion_Mayor_Menor (self, individuo, capacidades): #Sustituimos una base de un SD por otra -> Hasta cumplir restricción
        capacidades_sd = list(np.zeros(self.Num_Max))   #Capacidades de los SD
        suma_capacidades = list(np.zeros(self.Num_Max)) #Suma de las capacidades de las bases
        for i in range(self.Num_Max):
            indices_bases_reparar = [j for j, value in enumerate(individuo) if value == i]
            capacidades_sd_i = capacidades[indices_bases_reparar]
            capacidades_sd[i] = capacidades_sd_i
            suma_capacidades[i] = sum(capacidades_sd[i])    #Almacenamos todas las sumas de las capacidades en un array
        Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  #Comprobamos en qué SD's se han superado la capacidad
        if len(Caps_SD_Superadas) > 0:
            while True:
                k_2 = np.argsort(suma_capacidades)[::-1]
                k = k_2[0]  # Solucionamos aquella capacidad que sea mas grande
                while True:
                    k_3 = random.choice(k_2[len(suma_capacidades) - 4:len(suma_capacidades)])  # Jugamos con uno de los 4 SD con menos suma de bases
                    indices_bases_SD = [j for j, value in enumerate(individuo) if value == k]    #Obtenemos índices de las bases cuya suma de caps supera el umbral
                    indices_resto_bases = [j for j, value in enumerate(individuo) if value == k_3]  # Obtenemos índices del resto de bases
                    capacidades_bases_SD_ordenados = list(np.argsort([capacidades[i] for i in indices_bases_SD])[::-1])
                    indices_bases_SD_ordenados = [indices_bases_SD[i] for i in capacidades_bases_SD_ordenados]

                    if (suma_capacidades[k] > 200 and suma_capacidades[k] < 210) or (suma_capacidades[k] < 200 and suma_capacidades[k] > 190):
                        indice_base_1 = indices_bases_SD_ordenados[len(indices_bases_SD_ordenados)-np.random.randint(1,len(indices_bases_SD_ordenados))] #Cuando se estabilice la suma de capacidades cogemos caps pequeñas
                        lista_filtrada = [value for value in indices_resto_bases if capacidades[value] <= capacidades[indice_base_1]]
                        if lista_filtrada:
                            indice_base_aleatoria_2 = random.choice(lista_filtrada)  # Elección aleatoria de la base del resto de bases
                        else:
                            indice_base_aleatoria_2 = np.random.randint(0, numero_bases)
                            while True:
                                if indice_base_aleatoria_2 == indice_base_1:
                                    indice_base_aleatoria_2 = np.random.randint(0, numero_bases)
                                else:
                                    break
                    else:
                        indice_base_1 = indices_bases_SD_ordenados[0]
                    #indice_base_1 = random.choice(indices_bases_SD_ordenados[0:3])  # Elegimos una de las 5 bases del SD con mayor capacidad
                        lista_filtrada = [value for value in indices_resto_bases if capacidades[value] < capacidades[indice_base_1]]
                        if lista_filtrada:
                            indice_base_aleatoria_2 = random.choice(lista_filtrada)  # Elección aleatoria de la base del resto de bases
                        else:
                            indice_base_aleatoria_2 = np.random.randint(0, numero_bases)
                            while True:
                                if indice_base_aleatoria_2 == indice_base_1:
                                    indice_base_aleatoria_2 = np.random.randint(0, numero_bases)
                                else:
                                    break
                    if abs(200 - suma_capacidades[k_2[9]]) < 20 and suma_capacidades[k_2[9]] < 200:
                        individuo[indice_base_1], individuo[indice_base_aleatoria_2] = individuo[indice_base_aleatoria_2], individuo[indice_base_1]  # Intercambio posiciones de las bases
                    else:
                        e = random.randint(0, 5)
                        f = indices_bases_SD_ordenados[0:e]
                        individuo[f] = k_2[9]
                    F = np.array([t for t, x in enumerate(individuo) if x == k], dtype=int)
                    I = np.array(capacidades)
                    suma_capacidades[k] = sum(I[F])
                    if suma_capacidades[k] > 200:
                        continue
                    else:
                        break

                for i in range(numero_supply_depots):   #Bucle para comprobar sumas de capacidades alrededor de un SD
                    F = np.array([t for t, x in enumerate(individuo) if x == i], dtype=int)
                    I = np.array(capacidades)
                    suma_capacidades[i] = sum(I[F])
                Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad
                if len(Caps_SD_Superadas) == 0:
                    break
        return individuo

def Puntos_Sin_Repetir(num_points, offset=0.5):
    with rasterio.open(mapa_dem) as dem:
        limites = dem.bounds
    points = set()  # Usamos un conjunto para evitar duplicados
    while len(points) < num_points:
        latitud = np.random.uniform(low=limites.bottom, high=limites.top)
        longitud = np.random.uniform(low=limites.left, high=limites.right)
        # Aplicar desplazamiento aleatorio para evitar superposiciones
        latitud_offset = np.random.uniform(low=-offset, high=offset)
        longitud_offset = np.random.uniform(low=-offset, high=offset)
        point_with_offset = (longitud + longitud_offset, latitud + latitud_offset)
        points.add(point_with_offset)  # Agregamos el punto al conjunto
    return points

def Distancia_Base_Supply_Depot_2D(base, supply):
    if isinstance(base, list) and isinstance(supply, list):  # Cálculo de todas las distancias de bases e inters a SDs
        x_supply, y_supply = zip(*supply)
        x_base, y_base = zip(*base)
        dist = []
        for i in range(len(supply)):
            dist_aux = []
            for j in range(len(base)):
                distancia = math.sqrt((x_base[j] - x_supply[i]) ** 2 + (y_base[j] - y_supply[i]) ** 2)
                dist_aux.append(distancia)
            dist.append(dist_aux)
    else:  # Cálculo de distancia de una base al inter
        x_supply, y_supply = supply
        x_base, y_base = base
        dist = math.sqrt((x_base - x_supply) ** 2 + (y_base - y_supply) ** 2)
    return dist
def Funcion_Fitness(distancias, poblacion):
    lista_fitness = []
    for i in range(len(poblacion)):    #Aplicamos la función fitness a cada solución
        fitness = 0
        for j in range(numero_bases):
            SD = poblacion[i][j]    #Saco el SD asociado a una base de la población
            fitness += distancias[SD][j]    #Calculo fitness buscando en la matriz de distancias la distancia asociada
        fitness = fitness/numero_bases
        lista_fitness.append(fitness)
    return lista_fitness

def GetAltura(long, lat, dem):
    limites = dem.bounds  # Extraemos límites del mapa para hacer la matriz de superficie (Vienen en coordenadas espaciales)
    pixel_width = (limites.right - limites.left) / dem.width
    pixel_height = (limites.top - limites.bottom) / dem.height
    transform = from_origin(limites.left, limites.top, pixel_width, pixel_height)
    long, lat = ~transform * (long, lat)
    if 0 <= long < dem.width and 0 <= lat < dem.height:
        altura = dem.read(1)[int(lat), int(long)]    #Leemos el archivo para obtener la altura de las coordenadas especificadas
        return altura
    #Nota ->    x = Columna;    y = Fila;

def InterpolarPuntos(base, supply, num_puntos):
    x_base_2, y_base_2 = base
    x_supply_2, y_supply_2 = supply
    x_bases = np.linspace(x_base_2, x_supply_2, num_puntos)
    y_bases = np.linspace(y_base_2, y_supply_2, num_puntos)
    return x_bases, y_bases

def UTM_Geo(easting, northing): #Función para transformar coordenadas de UTM a Geográficas para distancias geodésicas
    transform_utm = Transformer.from_crs(crs_utm, crs_wgs84)
    long, lat = transform_utm.transform(easting,northing)
    return lat, long

def Representacion(Superficie,DistGrid):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = np.arange(0, dem.width, DistGrid)
    Y = np.arange(0, dem.height, DistGrid)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(Superficie)
    Z = np.transpose(Z)
    surf=ax.plot_surface(X, Y, Z,cmap='jet', rcount=150, ccount=150, edgecolor='none')
    fig.colorbar(surf)
    ax.set_xlabel("Distancia (m)")
    ax.set_ylabel("Distancia (m)")
    plt.show()

def Distancia_Base_Supply_Depot_3D(base,supply):    #Bases y SDs como coordenadas UTM
    dist = []
    for i in range(len(supply)):  # Para cada SD calculamos la distancia en 3D a todas las bases
        dist_aux = []
        for j in range(len(base)):
            # Para calcular distancias en 3D -> Sacamos muchos puntos entre base y SD
            # Calculamos distancias entre los segmentos formados entre cada uno de esos puntos y vamos sumándolas
            # La distancia la calcularemos teniendo en cuenta la distancia geodésica, esto es, teniendo en cuenta la curvatura del segmento

            x_puntos, y_puntos = InterpolarPuntos(base[j], supply[i], puntos_interpolado)  # Puntos para segmentos
            altura_puntos = [GetAltura(x, y, dem) for x, y in zip(x_puntos, y_puntos)]  # Sacamos la altura de los puntos interpolados
            distancia = 0.0
            for k in range(len(x_puntos)-1):  #Bucle para hacer el cálculo de distancias
                y_aux, x_aux = UTM_Geo(x_puntos[k], y_puntos[k])
                y_aux_2, x_aux_2 = UTM_Geo(x_puntos[k+1], y_puntos[k+1])
                punto1 = (y_aux, x_aux)
                punto2 = (y_aux_2, x_aux_2)
                distancia_x = geodesic(punto1, punto2).meters   #Distancia geodésica entre los dos puntos
                distancia_y = altura_puntos[k+1] - altura_puntos[k]
                distancia_segmento = math.sqrt((distancia_x ** 2) + (distancia_y ** 2))
                distancia += distancia_segmento
            dist_aux.append(distancia)
        dist.append(dist_aux)
    return dist



if __name__ == "__main__":
    # Definicion de los parámetros del genético
    Num_Individuos = 100
    Num_Generaciones = 10
    Tam_Individuos = 200
    Prob_Padres = 0.1
    Prob_Mutacion = 0.01
    Prob_Cruce = 0.5

    mapa_dem = 'PNOA_MDT05_ETRS89_HU30_0560_LID.tif'
    puntos_interpolado = 25  # Necesarios para calcular la distancia entre puntos en el mapa en 3D
    distGrid = 1
    # Definir el sistema de coordenadas UTM y WGS84
    crs_utm = CRS.from_epsg(25830)  # EPSG:25830 es UTM zona 30N, ETRS89 (Sistema de referencia geodésica para Europa, propio de este tipo de UTM [EPSG:25830])
    crs_wgs84 = CRS.from_epsg(4326) # EPSG:4326 es WGS84 (Sistema de referencia geodésico compatible con ETRS89)

    Pob_Actual = []
    Costes = []
    numero_bases = 200
    numero_supply_depots = 10
    capacidad_maxima = 20
    Ruta_Puntos = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT',
        f"Bases_SD.csv")
    Ruta_Capacidades = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT',
        f"Cap_Bases_SD.csv")
    if not os.path.exists(Ruta_Puntos):
        puntos = list(Puntos_Sin_Repetir(numero_bases + numero_supply_depots))
        puntos = np.array(puntos)
        np.savetxt(Ruta_Puntos, puntos, delimiter=',')
    else:
        puntos = []
        with open(Ruta_Puntos, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                numbers = tuple(numbers)
                puntos.append(numbers)
    if not os.path.exists(Ruta_Capacidades):
        capacidad_bases = np.random.randint(1, capacidad_maxima, size=(numero_bases))
        np.savetxt(Ruta_Capacidades, capacidad_bases, delimiter=',')
    else:
        capacidad_bases = []
        with open(Ruta_Capacidades, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = float(fila[0])
                capacidad_bases.append(int(numbers))
            capacidad_bases = np.array(capacidad_bases)
    supply_depots = puntos[-numero_supply_depots:]
    bases = puntos[:numero_bases]
    longitudes_bases, latitudes_bases = zip(*bases)
    longitudes_bases, latitudes_bases = list(longitudes_bases), list(latitudes_bases)
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    longitudes_supply_depots, latitudes_supply_depots = zip(*supply_depots)
    longitudes_supply_depots, latitudes_supply_depots = list(longitudes_supply_depots), list(latitudes_supply_depots)
    capacidad_supply_depots = np.full(numero_supply_depots,200)


    # distancias_euclideas = Distancia_Base_Supply_Depot_2D(bases, supply_depots) #Obtenemos distancias de bases a supply depots

    #Leemos el mapa DEM -> La primera banda, ya que suele tener datos de elevaciones
    with rasterio.open(mapa_dem) as dem:
        dem_data = dem.read(1)  # Leer la primera banda
        distancias_Oro = os.path.join(
            r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT',
            f"dist_Oro.csv")
        if not os.path.exists(distancias_Oro):
            distancias_3D = Distancia_Base_Supply_Depot_3D(bases, supply_depots)
            distancias_3D = np.array(distancias_3D)
            np.savetxt(distancias_Oro, distancias_3D, delimiter=',')
        else:
            distancias_3D = []
            with open(distancias_Oro, mode='r') as file:
                csv_reader = csv.reader(file)
                for fila in csv_reader:
                    # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                    numbers = [float(x) for x in fila]
                    distancias_3D_aux = []
                    for i in range(len(numbers)):
                        distancias_3D_aux.append(numbers[i])
                    distancias_3D.append(distancias_3D_aux)
                distancias_3D = np.array(distancias_3D)
        eje_x = np.arange(0, dem.width, distGrid)
        eje_y = np.arange(0, dem.height, distGrid)
        matriz_superficie = np.zeros(len(eje_x),len(eje_y))   #Matriz de superficie
        for i in range(len(eje_x)):
            for j in range(len(eje_y)):
                matriz_superficie[i, j] = GetAltura(eje_x[i], eje_y[i], dem)  #Rellenamos la matriz de superficie con las alturas
        Representacion(matriz_superficie, distGrid)



        ### A CONTINUACIÓN, APLICAMOS EL ALGORITMO DESPUÉS DE OBTENER LOS COSTES Y DISTANCIAS
    
    Ev1 = EvolutiveClass(Num_Individuos, Num_Generaciones, Tam_Individuos,numero_supply_depots, Prob_Padres, Prob_Mutacion, Prob_Cruce)
    #Ev1.ImprimirInformacion()
    Pob_Inicial = Ev1.PoblacionInicial(capacidad_bases, 100, numero_bases, numero_supply_depots,)  #Poblacion inicial -> 100 posibles soluciones -> PADRES
    for i in range(Num_Generaciones):
        print(("Generación: " + str(i + 1)))
        Pob_Actual = Ev1.Cruce(Pob_Inicial, capacidad_bases, numero_supply_depots)   #Aplicamos cruce en las soluciones
        Fitness = Funcion_Fitness(distancias_3D, Pob_Actual)
        Pob_Inicial, Costes = Ev1.Seleccion(Pob_Actual,Fitness)
    Sol_Final = Pob_Inicial[0]   #La primera población será la que tenga menor coste
    Coste_Final = Costes[0]
    print("Solución final:")
    for j in range(Tam_Individuos):
        print("Base " + str(j) + "-> SD: " + str(Sol_Final[j]))
    print("Coste de la solución: " + str(Coste_Final))


    # Graficar el mapa y los puntos
    dem_data = np.where(dem_data == dem.nodata, np.nan, dem_data)
    plt.figure(figsize=(10, 6))
    plt.imshow(dem_data, cmap='terrain')
    plt.colorbar(label='Altura (m)')
    plt.show()
    #plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    #plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    #for k in range(Tam_Individuos):
    #    plt.plot([longitudes_bases[k],longitudes_supply_depots[Sol_Final[k]]], [latitudes_bases[k], latitudes_supply_depots[Sol_Final[k]]],color='red')
    #plt.xlabel('Longitud')
    #plt.ylabel('Latitud')
    #plt.title('Mapa con Puntos Aleatorios')
    #plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
