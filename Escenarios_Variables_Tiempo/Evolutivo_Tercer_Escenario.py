# -*- coding: utf-8 -*-
"""
@author: Sergio

1º -> Activamos el entorno de conda -> conda activate TFM_MUIT
2º -> Podemos ejecutar el código sin problemas
"""
import rasterio
from rasterio.transform import from_origin
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
    points = set()  # Usamos un conjunto para evitar duplicados
    with rasterio.open(mapa_dem) as dem:
        while len(points) < num_points:
            latitud = np.random.uniform(low=0, high=dem.height)
            longitud = np.random.uniform(low=0, high=dem.width)
            # Aplicar desplazamiento aleatorio para evitar superposiciones
            latitud_offset = np.random.uniform(low=-offset, high=offset)
            longitud_offset = np.random.uniform(low=-offset, high=offset)
            point_with_offset = (latitud + latitud_offset, longitud + longitud_offset)
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

def GetAltura(lat, long, dem):
    limites = dem.bounds  # Extraemos límites del mapa para hacer la matriz de superficie (Vienen en coordenadas espaciales)
    pixel_width = (limites.right - limites.left) / dem.width
    pixel_height = (limites.top - limites.bottom) / dem.height
    transform = from_origin(limites.left, limites.top, pixel_width, pixel_height)
    columna, fila = ~transform * (long, lat)
    if 0 <= columna < dem.width and 0 <= fila < dem.height:
        altura = dem.read(1)[int(fila), int(columna)]    #Leemos el archivo para obtener la altura de las coordenadas especificadas
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
    Y, X = np.meshgrid(X, Y)
    Z = np.array(Superficie)
    surf=ax.plot_surface(X, Y, Z,cmap='jet', rcount=150, ccount=150, edgecolor='none')
    fig.colorbar(surf)
    plt.show()

def Distancia_Base_Supply_Depot_3D(base,supply, dem):    #Bases y SDs como coordenadas UTM
    if isinstance(base, list) and isinstance(supply, list): #Para calcular distancias de bases a SD
        dist = []
        for i in range(len(supply)):  # Para cada SD calculamos la distancia en 3D a todas las bases
            dist_aux = []
            for j in range(len(base)):
            # Para calcular distancias en 3D -> Sacamos muchos puntos entre base y SD
            # Calculamos distancias entre los segmentos formados entre cada uno de esos puntos y vamos sumándolas
            # La distancia la calcularemos teniendo en cuenta la distancia geodésica, esto es, teniendo en cuenta la curvatura del segmento

                #Antes comprobaremos si la trayectoria de una base al SD intersecta con algún tramo del río
                if InterseccionRectas(base[j], supply[i], puntos_rio_aux):  #Vienen en coordenadas UTM
                    contador = 0
                    distancia = 0.0
                    distancia_rio_2 = 0.0
                    for k in range(2):  #Tenemos que calcular 2 distancias -> De la base al muelle y del muelle al SD
                        if contador == 0:
                            y_puntos, x_puntos = InterpolarPuntos(base[j], puente_rio_UTM,puntos_interpolado)  # Puntos para segmentos
                        else:
                            y_puntos, x_puntos = InterpolarPuntos(supply[i], puente_rio_UTM,puntos_interpolado)  # Puntos para segmentos
                        altura_puntos = [GetAltura(y, x, dem) for y, x in zip(y_puntos, x_puntos)]  # Sacamos la altura de los puntos interpolados
                        distancia_rio = 0.0
                        for k in range(len(x_puntos) - 1):  # Bucle para hacer el cálculo de distancias
                            y_aux, x_aux = UTM_Geo(x_puntos[k], y_puntos[k])
                            y_aux_2, x_aux_2 = UTM_Geo(x_puntos[k + 1], y_puntos[k + 1])
                            punto1 = (y_aux, x_aux)
                            punto2 = (y_aux_2, x_aux_2)
                            distancia_x = geodesic(punto1, punto2).meters  # Distancia geodésica entre los dos puntos
                            distancia_y = altura_puntos[k + 1] - altura_puntos[k]
                            distancia_segmento = math.sqrt((distancia_x ** 2) + (distancia_y ** 2))
                            distancia_rio += distancia_segmento
                        distancia_rio_2 += distancia_rio
                        contador += 1
                    distancia += distancia_rio_2
                else:
                    y_puntos, x_puntos = InterpolarPuntos(base[j], supply[i], puntos_interpolado)  # Puntos para segmentos
                    altura_puntos = [GetAltura(y, x, dem) for y, x in zip(y_puntos, x_puntos)]  # Sacamos la altura de los puntos interpolados
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
    else:   #Para calcular distancias entre puntos (Río)
        dist = []
        y_puntos, x_puntos = InterpolarPuntos(base, supply, puntos_interpolado)  # Puntos para segmentos
        altura_puntos = [GetAltura(y, x, dem) for y, x in zip(y_puntos, x_puntos)]  # Sacamos la altura de los puntos interpolados
        distancia = 0.0
        for k in range(len(x_puntos) - 1):  # Bucle para hacer el cálculo de distancias
            y_aux, x_aux = UTM_Geo(x_puntos[k], y_puntos[k])
            y_aux_2, x_aux_2 = UTM_Geo(x_puntos[k + 1], y_puntos[k + 1])
            punto1 = (y_aux, x_aux)
            punto2 = (y_aux_2, x_aux_2)
            distancia_x = geodesic(punto1, punto2).meters  # Distancia geodésica entre los dos puntos
            distancia_y = altura_puntos[k + 1] - altura_puntos[k]
            distancia_segmento = math.sqrt((distancia_x ** 2) + (distancia_y ** 2))
            distancia += distancia_segmento
        dist.append(distancia)
        return dist

def InterseccionRectas(base,supply,rio):
    contador_rio = 0
    for i in range(len(rio)-1):
        o1 = OrientacionRectas(base, supply, rio[i])
        o2 = OrientacionRectas(base, supply, rio[i+1])
        o3 = OrientacionRectas(rio[i], rio[i+1], base)
        o4 = OrientacionRectas(rio[i], rio[i+1], supply)
        if o1 != o2 and o3 != o4:   #Si para una recta las orientaciones son distintas y para la otra también, quiere decir que se intersectan
            contador_rio += 1
    if (contador_rio % 2) == 0: #Si número de veces que se cruza con el río es par -> BASE Y SD EN EL MISMO LADO DEL RÍO
        return False
    else:                       #Si número de veces que se cruza con el río es impar -> BASE Y SD EN LADOS OPUESTOS DEL RÍO
        return True

def OrientacionRectas(p1,p2,p3):    #Función para determinar si un punto está de un lado o de otro de una recta
    orient = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
    if orient == 0: #Si da 0, quiere decir que ese punto es colineal -> No está en un lado ni en otro
        return 0
    if orient > 0:  #Si está a un lado
        return 1
    if orient < 0:  #Si está al otro
        return 2

if __name__ == "__main__":
    # Definicion de los parámetros del genético
    random.seed(2030)
    np.random.seed(2030)
    Num_Individuos = 100
    Num_Generaciones = 300
    Tam_Individuos = 200
    Prob_Padres = 0.5
    Prob_Mutacion = 0.01
    Prob_Cruce = 0.5

    mapa_dem = 'PNOA_MDT05_ETRS89_HU30_0560_LID.tif'
    puntos_interpolado = 50  # Necesarios para calcular la distancia entre puntos en el mapa en 3D
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
        r'.\Resultados\Orografia',
        f"Bases_SD_1.csv")
    Ruta_Capacidades = os.path.join(
        r'.\Resultados\Orografia',
        f"Cap_Bases_SD_1.csv")
    distancias_Oro = os.path.join(
        r'.\Escenarios_Variables_Tiempo\Distancias_Mapas',
        f"dist_Tercer_1.csv")
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
    latitudes_bases_2, longitudes_bases = zip(*bases)
    latitudes_bases = np.array(latitudes_bases_2)
    with rasterio.open(mapa_dem) as dem:  # Transformamos a UTM de la zona del mapa para posteriores operaciones
        limites = dem.bounds
        latitudes_bases = abs(latitudes_bases - dem.height)
        pixel_width = (limites.right - limites.left) / dem.width
        pixel_height = (limites.top - limites.bottom) / dem.height
        transform = from_origin(limites.left, limites.top, pixel_width, pixel_height)
    bases_UTM = []
    for i in range(len(longitudes_bases)):
        lon, lat = transform * (longitudes_bases[i], latitudes_bases[i])
        bases_UTM.append((lat, lon))
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    latitudes_supply_depots_2, longitudes_supply_depots = zip(*supply_depots)
    latitudes_supply_depots = np.array(latitudes_supply_depots_2)
    latitudes_supply_depots = abs(latitudes_supply_depots - dem.height)
    SD_UTM = []
    for i in range(len(longitudes_supply_depots)):
        lon, lat = transform * (longitudes_supply_depots[i], latitudes_supply_depots[i])
        SD_UTM.append((lat, lon))
    capacidad_supply_depots = np.full(numero_supply_depots,200)

    puntos_rio = np.array([[1890,3710], [1945,3214], [1808,2406], [2193,1625], [2588,1019], [2864, 744], [3029, 450], [2699,119]])
    long_rio_total = []
    distancias_Rio = os.path.join(
        r'.\Escenarios_Variables_Tiempo\Distancias_Mapas',
        f"dist_Rio.csv")
    puntos_rio_aux = np.zeros(puntos_rio.shape)
    for i in range(puntos_rio.shape[0]):    #Los pasamos a UTM para hacer luego las intersecciones de cada tramo del río con las de bases y SD
        lon, lat = transform * (puntos_rio[i][0], puntos_rio[i][1])
        puntos_rio_aux[i] = [lat, lon]
    if not os.path.exists(distancias_Rio):
        with rasterio.open(mapa_dem) as dem:
            for i in range(puntos_rio.shape[0]-1):
                long_rio = Distancia_Base_Supply_Depot_2D(puntos_rio[i], puntos_rio[i+1])
                long_rio_total.append(long_rio)
            np.savetxt(distancias_Rio, long_rio_total, delimiter=',')
    else:
        long_rio_total = []
        with open(distancias_Rio, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                long_rio_total.append(numbers)
            long_rio_total = np.array(long_rio_total)

    punto_Rio = os.path.join(
        r'.\Escenarios_Variables_Tiempo\Distancias_Mapas',
        f"Muelle_Rio.csv")
    if not os.path.exists(punto_Rio):
        with rasterio.open(mapa_dem) as dem:
            rand_distancia = np.random.uniform(0,np.sum(long_rio_total))   #Distancia aleatoria en la que estará el punto que actúe como puente
            rio_acum = np.cumsum(long_rio_total)  # Acumulamos las distancias entre cada par de puntos para ver la longitud total en cada tramo
            indice_tramo = np.searchsorted(rio_acum,rand_distancia)  # Determinamos en qué tramo del río está el punto generado

            if indice_tramo == 0:  # Sacamos la distancia que hay en el tramo seleccionado, donde irá el punto
                distancia_tramo = rand_distancia
            else:
                distancia_tramo = rand_distancia - rio_acum[indice_tramo - 1]

            inicio_tramo = np.array([puntos_rio[indice_tramo][1], puntos_rio[indice_tramo][0]])
            final_tramo = np.array([puntos_rio[indice_tramo + 1][1], puntos_rio[indice_tramo + 1][0]])
            vector_tramo = np.array([final_tramo[0] - inicio_tramo[0], final_tramo[1] - inicio_tramo[1]])
            vector_unitario_tramo = vector_tramo / np.linalg.norm(vector_tramo)  # Sacamos el vector unitario que conforma el tramo del río donde poner el punto

            puente_rio = inicio_tramo + vector_unitario_tramo * distancia_tramo  # Punto del río en el que habrá un puente, un muelle...
            lon, lat = transform * (puente_rio[0], puente_rio[1])
            puente_rio_UTM = np.array((lat, lon))
            np.savetxt(punto_Rio, puente_rio_UTM, delimiter=',')
    else:
        puente_rio_UTM = []
        with open(punto_Rio, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                puente_rio_UTM.append(numbers[0])
            puente_rio_UTM = tuple(puente_rio_UTM)

    # distancias_euclideas = Distancia_Base_Supply_Depot_2D(bases, supply_depots) #Obtenemos distancias de bases a supply depots

    #Leemos el mapa DEM -> La primera banda, ya que suele tener datos de elevaciones
    with rasterio.open(mapa_dem) as dem:
        dem_data = dem.read(1)  # Leer la primera banda
        dem_data = np.where(dem_data == dem.nodata, np.nan, dem_data)
        if not os.path.exists(distancias_Oro):
            distancias_3D = Distancia_Base_Supply_Depot_3D(bases_UTM, SD_UTM, dem)
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
        #eje_x = np.arange(0, dem.width, distGrid)
        #eje_y = np.arange(0, dem.height, distGrid)
        #X, Y = np.meshgrid(eje_x, eje_y)
        #matriz_superficie = dem_data[Y, X]   #Matriz de superficie
        Representacion(dem_data, distGrid)



        ### A CONTINUACIÓN, APLICAMOS EL ALGORITMO DESPUÉS DE OBTENER LOS COSTES Y DISTANCIAS
    
    Ev1 = EvolutiveClass(Num_Individuos, Num_Generaciones, Tam_Individuos,numero_supply_depots, Prob_Padres, Prob_Mutacion, Prob_Cruce)
    Costes_Generacion = []
    Pob_Inicial = Ev1.PoblacionInicial(capacidad_bases, 100, numero_bases, numero_supply_depots,)  #Poblacion inicial -> 100 posibles soluciones -> PADRES
    for i in range(Num_Generaciones):
        print(("Generación: " + str(i + 1)))
        Fitness = Funcion_Fitness(distancias_3D, Pob_Inicial)
        Pob_Actual, Costes = Ev1.Seleccion(Pob_Inicial, Fitness)
        Pob_Inicial = Ev1.Cruce(Pob_Actual, capacidad_bases, numero_supply_depots)  # Aplicamos cruce en las soluciones
        print("Coste: " + str(Costes[0]))
        Costes_Generacion.append(Costes[0])
    Sol_Final = Pob_Inicial[0]   #La primera población será la que tenga menor coste
    Coste_Final = Costes[0]
    print("Solución final:")
    for j in range(Tam_Individuos):
        print("Base " + str(j) + "-> SD: " + str(Sol_Final[j]))
    print("Coste de la solución: " + str(Coste_Final))

    bases_figure = np.transpose(np.array([latitudes_bases, longitudes_bases]))
    SD_figure = np.transpose(np.array([latitudes_supply_depots, longitudes_supply_depots]))
    # Graficar el mapa y los puntos
    fig_1 = plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases_2, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots_2, color='black', marker='p', s=60,label='Puntos de Suministro')
    fig_1.show()
    # Evolución del coste de una de las rutas
    coste = plt.figure(figsize=(10, 6))
    plt.plot(Costes_Generacion)
    plt.xlabel('Número de ejecuciones (Genético)')
    plt.ylabel('Distancia (px/m)')
    coste.show()
    #Mapa a usar
    #plt.figure(figsize=(10, 6))
    #plt.imshow(dem_data, cmap='terrain')
    #plt.colorbar(label='Altura (m)')
    #Generamos el río -> Lo hacemos observando el mapa y viendo dónde hay una mayor depresión del terreno
    #Dibujamos muelle
    lon, lat = ~transform * (puente_rio_UTM[1], puente_rio_UTM[0])
    puente_rio = np.array((lat, lon))
    #plt.show()
    # Graficamos solución
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases_2, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots_2, color='black', marker='p',s=60,label='Puntos de Suministro')
    for i in range(len(puntos_rio)-1):
        plt.plot([puntos_rio[i][1], puntos_rio[i+1][1]], [puntos_rio[i][0], puntos_rio[i+1][0]], color='black')
    plt.scatter(puente_rio[1], puente_rio[0], color='black', label='Muelle')
    plt.xlabel('Distancia Horizontal (px/m)')
    plt.ylabel('Distancia Vertical (px/m)')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    #Unimos rectas
    for k in range(numero_supply_depots):
        SD = [i for i,v in enumerate(Sol_Final) if v == k]  #Sacamos bases asociadas a un SD
        if len(SD) > 0: # Porque puede haber bases que no tengan asociado el SD de la iteración que toca
            aux = random.choice(SD)  # Base aleatoria
            if InterseccionRectas(bases_figure[aux], SD_figure[k], puntos_rio):
                plt.plot([longitudes_bases[aux],puente_rio[1]], [latitudes_bases_2[aux], puente_rio[0]],color='red')
                plt.plot([puente_rio[1],longitudes_supply_depots[Sol_Final[aux]]], [puente_rio[0], latitudes_supply_depots_2[Sol_Final[aux]]],color='red')
            else:
                plt.plot([longitudes_bases[aux],longitudes_supply_depots[Sol_Final[aux]]], [latitudes_bases_2[aux], latitudes_supply_depots_2[Sol_Final[aux]]],color='red')
    plt.gca().invert_yaxis()
    plt.show()
