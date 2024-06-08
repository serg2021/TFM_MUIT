import numpy as np
from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateReal import *
from PyCROSL.SubstrateInt import *
import random
import math
import matplotlib.pyplot as plt


class Fitness(AbsObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size    #Número de bases
        super().__init__(self.size, opt)

    def objective(self, solution):  #Función objetivo -> El algoritmo se encarga de hacerlo, no es como el evolutivo
        return Funcion_Fitness(distancias_euclideas, solution)

    def random_solution(self):  #Generamos una población inicial -> Solo indicamos cómo serán las soluciones y las reparamos una vez se generen, el resto lo hace el algoritmo
        Pob_Ini = np.random.randint(0, numero_supply_depots, size=self.size)  # Solución tipo
        if(Comprobacion_Individuo(Pob_Ini, capacidad_bases)):
            Pob_Ini = Reparacion_Mayor_Menor(Pob_Ini, capacidad_bases)
        return Pob_Ini

    def repair_solution(self, solution):    #Reparación de individuos
        if (Comprobacion_Individuo(solution, capacidad_bases)):
            solution = Reparacion_Mayor_Menor(solution, capacidad_bases)
            #Lo hemos reparado en base a la capacidad -> PERO NO EN BASE A LOS SD POSIBLES -> SALEN VALORES DE SD QUE NO SON
            #Tenemos que añadir AQUÍ una forma de repararlo -> Lo más sencillo es hacer un bucle que recorra cada elemento de la solución y
            #Cambiar esos valores por otros aleatorios que estén dentro del rango
        for i in range(numero_bases):
            if solution[i] > 9 or solution[i] < 0:
                solution[i] = np.random.randint(0, numero_supply_depots)
        return solution

def Puntos_Sin_Repetir(num_points, offset=0.5):
    points = set()  # Usamos un conjunto para evitar duplicados
    while len(points) < num_points:
        latitud = np.random.uniform(low=0, high=180.0)
        longitud = np.random.uniform(low=0, high=180.0)
        # Aplicar desplazamiento aleatorio para evitar superposiciones
        latitud_offset = np.random.uniform(low=-offset, high=offset)
        longitud_offset = np.random.uniform(low=-offset, high=offset)
        point_with_offset = (latitud + latitud_offset, longitud + longitud_offset)
        points.add(point_with_offset)  # Agregamos el punto al conjunto
    return points

def Distancia_Base_Supply_Depot_2D(base, supply):
    x_supply, y_supply = zip(*supply)
    x_base, y_base = zip(*base)
    dist = []
    for i in range(len(supply)):
        dist_aux = []
        for j in range(len(base)):
            distancia = math.sqrt((x_base[j]-x_supply[i])**2 + (y_base[j]-y_supply[i])**2)
            dist_aux.append(distancia)
        dist.append(dist_aux)
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
    for i in range(numero_supply_depots):
        indices_bases = [j for j, value in enumerate(individuo) if value == i]  #Obtenemos los indices de las bases asociadas a un SD "i"
        comprobar_capacidades = capacidades[indices_bases]
        if sum(comprobar_capacidades) > 200:    #Si la suma de las capacidades de una solucion para un supply depot es mayor que 200 -> REPARACION
            return True
        else:
            return False

def Reparacion_Mayor_Menor (individuo, capacidades): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
    capacidades_sd = list(np.zeros(numero_supply_depots))   #Capacidades de los SD
    suma_capacidades = list(np.zeros(numero_supply_depots)) #Suma de las capacidades de las bases
    for i in range(numero_supply_depots):
        indices_bases_reparar = [j for j, value in enumerate(individuo) if value == i]
        capacidades_sd_i = capacidades[indices_bases_reparar]
        capacidades_sd[i] = capacidades_sd_i
        suma_capacidades[i] = sum(capacidades_sd[i])    #Almacenamos todas las sumas de las capacidades en un array
    Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  #Comprobamos en qué SD's se han superado la capacidad
    for k in Caps_SD_Superadas:
        while True:
            indices_bases_SD = [j for j, value in enumerate(individuo) if value == k]    #Obtenemos índices de las bases cuya suma de caps supera elumbral
            indices_bases_SD = np.array(indices_bases_SD)
            indices_resto_bases = [j for j, value in enumerate(individuo) if value != k]  # Obtenemos índices del resto de bases
            capacidades_bases_SD_ordenados = list(np.argsort(capacidades[indices_bases_SD])[::-1])
            indices_bases_SD_ordenados = indices_bases_SD[capacidades_bases_SD_ordenados]

            indice_base_1 = indices_bases_SD_ordenados[0] #Elegimos la base del SD con mayor capacidad
            indice_base_aleatoria_2 = random.choice([value for value in indices_resto_bases if capacidades[value] < capacidades[indice_base_1]])  #Elección aleatoria de la base del resto de bases
            individuo[indice_base_1], individuo[indice_base_aleatoria_2] = individuo[indice_base_aleatoria_2], individuo[indice_base_1] #Intercambio posiciones de las bases
            indices_bases_reparadas = [j for j, value in enumerate(individuo) if value == k]    #Obtenemos índices de las bases cuya suma de caps supera el umbral
            if sum(capacidades[indices_bases_reparadas]) > 200: #Si es más de 200, volvemos a hacer mismo código del while
                continue
            else: #Si no, salimos del while y avanzamos en el for
                break
    return individuo

if __name__ == "__main__":

    Pob_Actual = []
    Costes = []
    poblacion_inicial = 100
    numero_bases = 200
    numero_supply_depots = 10
    capacidad_maxima = 20
    puntos = list(Puntos_Sin_Repetir(numero_bases+numero_supply_depots))
    supply_depots = puntos[-numero_supply_depots:]
    bases = puntos[:numero_bases]
    longitudes_bases, latitudes_bases = zip(*bases)
    capacidad_bases = np.random.randint(1, capacidad_maxima, size=(numero_bases))
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    longitudes_supply_depots, latitudes_supply_depots = zip(*supply_depots)
    capacidad_supply_depots = np.full(numero_supply_depots,200)

    distancias_euclideas = Distancia_Base_Supply_Depot_2D(bases, supply_depots) #Obtenemos distancias de bases a supply depots

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
    #Graficamos la solución
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    for k in range(numero_bases):
        plt.plot([longitudes_bases[k],longitudes_supply_depots[solution[k]]], [latitudes_bases[k], latitudes_supply_depots[solution[k]]],color='red')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()