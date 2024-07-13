import numpy as np
from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateInt import *
import random
import math
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS, Transformer
from geopy.distance import geodesic
import os
import csv


class Fitness(AbsObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size    #Número de bases
        super().__init__(self.size, opt)

    def objective(self, solution):  #Función objetivo -> El algoritmo se encarga de hacerlo, no es como el evolutivo
        return Funcion_Fitness(distancias_3D, solution)

    def random_solution(self):  #Generamos una población inicial -> Solo indicamos cómo serán las soluciones y las reparamos una vez se generen, el resto lo hace el algoritmo
        Pob_Ini = np.random.randint(0, numero_supply_depots, size=self.size)  # Solución tipo
        if(Comprobacion_Individuo(Pob_Ini, capacidad_bases)):
            Pob_Ini = Reparacion_Mayor_Menor(Pob_Ini, capacidad_bases)
        return Pob_Ini

    def repair_solution(self, solution):    #Reparación de individuos
        for i in range(numero_bases):
            if solution[i] > 9 or solution[i] < 0:
                solution[i] = np.random.randint(0, numero_supply_depots)
        if (Comprobacion_Individuo(solution, capacidad_bases)):
            solution = Reparacion_Mayor_Menor(solution, capacidad_bases)
            #Lo hemos reparado en base a la capacidad -> PERO NO EN BASE A LOS SD POSIBLES -> SALEN VALORES DE SD QUE NO SON
            #Tenemos que añadir AQUÍ una forma de repararlo -> Lo más sencillo es hacer un bucle que recorra cada elemento de la solución y
            #Cambiar esos valores por otros aleatorios que estén dentro del rango
        return solution

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
def Funcion_Fitness(distancias, individuo):
    fitness = 0
    for j in range(len(individuo)):
        SD = individuo[j]    #Saco el SD asociado a una base de la población
        if(SD > 9 or SD < 0 or isinstance(SD, float)):   #Está mutando y nos da valores de SD que no pueden ser -> SOLUCIÓN:
            SD = np.random.randint(0,numero_supply_depots)                                   # Se genera el número a modificar
        fitness += distancias[SD][j]    #Calculo fitness buscando en la matriz de distancias la distancia asociada
    fitness = fitness/numero_bases
    return fitness

def Comprobacion_Individuo (individuo, capacidades):
    suma_comprobar = list(np.zeros(numero_supply_depots))
    for i in range(numero_supply_depots):
        indices_bases = [j for j, value in enumerate(individuo) if value == i]  # Obtenemos los indices de las bases asociadas a un SD "i"
        comprobar_capacidades = capacidades[indices_bases]
        suma_comprobar[i] = sum(comprobar_capacidades)
        Caps_Comprobar = [ind_cap for ind_cap, j in enumerate(suma_comprobar) if j > 200]
        if len(Caps_Comprobar) > 0:
            return True

def Reparacion_Mayor_Menor (individuo, capacidades): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
    capacidades_sd = list(np.zeros(numero_supply_depots))  # Capacidades de los SD
    suma_capacidades = list(np.zeros(numero_supply_depots))  # Suma de las capacidades de las bases
    for i in range(numero_supply_depots):
        indices_bases_reparar = [j for j, value in enumerate(individuo) if value == i]
        capacidades_sd_i = capacidades[indices_bases_reparar]
        capacidades_sd[i] = capacidades_sd_i
        suma_capacidades[i] = sum(capacidades_sd[i])  # Almacenamos todas las sumas de las capacidades en un array
    Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad
    if len(Caps_SD_Superadas) > 0:
        while True:
            k_2 = np.argsort(suma_capacidades)[::-1]
            k = k_2[0]  # Solucionamos aquella capacidad que sea mas grande
            while True:
                k_3 = random.choice(k_2[len(suma_capacidades) - 4:len(suma_capacidades)])  # Jugamos con uno de los 4 SD con menos suma de bases
                indices_bases_SD = [j for j, value in enumerate(individuo) if value == k]  # Obtenemos índices de las bases cuya suma de caps supera el umbral
                indices_resto_bases = [j for j, value in enumerate(individuo) if value == k_3]  # Obtenemos índices del resto de bases
                capacidades_bases_SD_ordenados = list(np.argsort([capacidades[i] for i in indices_bases_SD])[::-1])
                indices_bases_SD_ordenados = [indices_bases_SD[i] for i in capacidades_bases_SD_ordenados]

                if (suma_capacidades[k] > 200 and suma_capacidades[k] < 210) or (suma_capacidades[k] < 200 and suma_capacidades[k] > 190):
                    indice_base_1 = indices_bases_SD_ordenados[len(indices_bases_SD_ordenados) - np.random.randint(1,len(indices_bases_SD_ordenados))]  # Cuando se estabilice la suma de capacidades cogemos caps pequeñas
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
                if abs(200 - suma_capacidades[k_2[9]]) < 50 and suma_capacidades[k_2[9]] < 200:
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

            for i in range(numero_supply_depots):  # Bucle para comprobar sumas de capacidades alrededor de un SD
                F = np.array([t for t, x in enumerate(individuo) if x == i], dtype=int)
                I = np.array(capacidades)
                suma_capacidades[i] = sum(I[F])
            Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad
            if len(Caps_SD_Superadas) == 0:
                break
    return individuo

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
    dist = []
    for i in range(len(supply)):  # Para cada SD calculamos la distancia en 3D a todas las bases
        dist_aux = []
        for j in range(len(base)):
            # Para calcular distancias en 3D -> Sacamos muchos puntos entre base y SD
            # Calculamos distancias entre los segmentos formados entre cada uno de esos puntos y vamos sumándolas
            # La distancia la calcularemos teniendo en cuenta la distancia geodésica, esto es, teniendo en cuenta la curvatura del segmento

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

if __name__ == "__main__":

    mapa_dem = 'PNOA_MDT05_ETRS89_HU30_0560_LID.tif'
    puntos_interpolado = 25  # Necesarios para calcular la distancia entre puntos en el mapa en 3D
    distGrid = 1
    # Definimos el sistema de coordenadas UTM y WGS84
    crs_utm = CRS.from_epsg(25830)  # EPSG:25830 es UTM zona 30N, ETRS89 (Sistema de referencia geodésica para Europa, propio de este tipo de UTM [EPSG:25830])
    crs_wgs84 = CRS.from_epsg(4326)  # EPSG:4326 es WGS84 (Sistema de referencia geodésico compatible con ETRS89)

    Pob_Actual = []
    Costes = []
    poblacion_inicial = 100
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
    latitudes_bases, longitudes_bases = zip(*bases)
    with rasterio.open(mapa_dem) as dem:
        limites = dem.bounds  # Extraemos límites del mapa para hacer la matriz de superficie (Vienen en coordenadas espaciales)
        pixel_width = (limites.right - limites.left) / dem.width
        pixel_height = (limites.top - limites.bottom) / dem.height
        transform = from_origin(limites.left, limites.top, pixel_width, pixel_height)
    bases_UTM = []
    for i in range(len(longitudes_bases)):
        lon, lat = transform * (longitudes_bases[i], latitudes_bases[i])
        bases_UTM.append((lat, lon))
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    latitudes_supply_depots, longitudes_supply_depots = zip(*supply_depots)
    SD_UTM = []
    for i in range(len(longitudes_supply_depots)):
        lon, lat = transform * (longitudes_supply_depots[i], latitudes_supply_depots[i])
        SD_UTM.append((lat, lon))
    capacidad_supply_depots = np.full(numero_supply_depots,200)

    #distancias_euclideas = Distancia_Base_Supply_Depot_2D(bases, supply_depots) #Obtenemos distancias de bases a supply depots

    #Leemos el mapa DEM -> La primera banda, ya que suele tener datos de elevaciones
    with rasterio.open(mapa_dem) as dem:
        dem_data = dem.read(1)  # Leer la primera banda
        distancias_Oro = os.path.join(
            r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT',
            f"dist_Oro.csv")
        if not os.path.exists(distancias_Oro):
            distancias_3D = Distancia_Base_Supply_Depot_3D(bases, supply_depots, dem)
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

    ## Hasta aquí generamos las bases y SD que vamos a tener antes del algoritmo -> Junto con las distancias asociadas

    objfunc = Fitness(numero_bases)  #Función objetivo para un tamaño de vector igual al número de bases
    params = {                      #Hiperparámetros del algoritmo
        "popSize": poblacion_inicial, #Población inicial
        "rho": 0.6, #Porcentaje de ocupación de corales del Reef inicial
        "Fb": 0.98, #Proporción de Broadcast Spawning
        "Fd": 0.2,  #Proporción de Depredación
        "Pd": 0.8,  #Probabilidad de Depredación
        "k": 3, #Número máximo de intentos para que la larva intente asentarse
        "K": 20,    #Número máximo de corales con soluciones duplicadas
        "group_subs": True, #Si 'True', los corales se reproducen sólo en su mismo substrato, si 'False', se reproducen con toda la población

        "stop_cond": "Ngen",   #Condición de parada
        "time_limit": 4000.0,   #Tiempo límite (real, no de CPU) de ejecución
        "Ngen": 100,  #Número de generaciones
        "Neval": 3e3,   #Número de evaluaciones de la función objetivo
        "fit_target": 50,   #Valor de función objetivo a alcanzar -> Ponemos 50 por poner un valor muy bajo

        "verbose": True,    #Informe periódico de cómo va el algoritmo
        "v_timer": 1,   #Tiempo entre informes generados
        "Njobs": 1, #Número de trabajos a ejecutar en paralelo -> Como es 1, se ejecuta de forma secuencial

        "dynamic": True,    #Determina si usar la variante dinámica del algoritmo -> Permite cambiar el tamaño de cada substrato (Mirar paper)
        "dyn_method": "success",    #Determina la probabilidad de elegir un substrato para cada coral en la siguiente generación -> Con 'success' usa el ratio de larvas exitosas en cada generación
        "dyn_metric": "best",    #Determina cómo agregar los valores de cada substrato para obtener la medida de cada uno
        "dyn_steps": 10,    #Número de evaluaciones por cada substrato
        "prob_amp": 0.001    #Determina cómo las diferencias entre las métricas de los substratos afectan la probabilidad de cada una -> Cuanto más pequeña, más amplifica
    }

    operators = [
        SubstrateInt("MutSample", {"method": "Gauss", "F": 1, "N": 3}),  # Rand Mutation -> F = Desviación Típica; N = Número de muestras a mutar
        SubstrateInt("Multipoint"),    #Multi-Point Crossover
        SubstrateInt("BLXalpha", {"F": 0.5}),  #BLX-Alpha -> F = Alpha
        SubstrateInt("DE/best/1", {"F": 0.7, "Cr": 0.8})   #Differential Evolution -> F = Factor de escalado de la ecuación; Cr = Prob. de Recombinación
    ]

    Coral = CRO_SL(objfunc,operators,params)
    solution, obj_value =Coral.optimize()

    print("Solución final:")
    for j in range(numero_bases):
        print("Base " + str(j) + "-> SD: " + str(solution[j]))
    print("Coste final: " + str(obj_value))
    # Graficar el mapa y los puntos
    dem_data = np.where(dem_data == dem.nodata, np.nan, dem_data)
    plt.figure(figsize=(10, 6))
    plt.imshow(dem_data, cmap='terrain')
    plt.colorbar(label='Altura (m)')
    plt.scatter(longitudes_bases, latitudes_bases, color='white', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    for k in range(numero_supply_depots):
        SD = [i for i,v in enumerate(solution) if v == k]  #Sacamos bases asociadas a un SD
        if len(SD) > 0: # Porque puede haber bases que no tengan asociado el SD de la iteración que toca
            aux = random.choice(SD)  # Base aleatoria
            plt.plot([longitudes_bases[aux],longitudes_supply_depots[solution[aux]]], [latitudes_bases[aux], latitudes_supply_depots[solution[aux]]],color='red')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()
    #Graficamos la solución