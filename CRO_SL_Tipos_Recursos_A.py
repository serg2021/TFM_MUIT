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
        for j in range(len(indices_bases)):
            if np.array_equal(lista_clases_base[indices_bases[j]], lista_clases_SD[i]):  # Comprobamos si tienen los mismos recursos la base asignada y el SD
                comprobar_capacidades = capacidades[indices_bases[j]]
                suma_comprobar[i] += comprobar_capacidades
            else:  # Si no, hay que reparar
                return True
    Caps_Comprobar = [ind_cap for ind_cap, j in enumerate(suma_comprobar) if j > 200]
    if len(Caps_Comprobar) > 0:
        return True

def Reparacion_Mayor_Menor (individuo, capacidades): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
    capacidades_sd = list(np.zeros(numero_supply_depots))  # Capacidades de los SD
    suma_capacidades = list(np.zeros(numero_supply_depots))  # Suma de las capacidades de las bases
    for i in range(numero_supply_depots):
        indices_bases_reparar = [j for j, value in enumerate(individuo) if value == i]
        for j in range(len(indices_bases_reparar)):
            if np.array_equal(lista_clases_base[indices_bases_reparar[j]],lista_clases_SD[i]):  # Comprobamos si tienen los mismos recursos la base asignada y el SD
                capacidades_sd_i = capacidades[indices_bases_reparar]
                capacidades_sd[i] = capacidades_sd_i
                suma_capacidades[i] = sum(capacidades_sd[i])  # Almacenamos todas las sumas de las capacidades en un array
            else:
                individuo[indices_bases_reparar[j]] = random.choice([k for k, v in enumerate(lista_clases_SD) if np.array_equal(v, lista_clases_base[indices_bases_reparar[j]])])
                # Asociamos a la base que queremos reparar un SD que coincida con sus recursos
    Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad
    if len(Caps_SD_Superadas) > 0:
        while True:
            k_2 = np.argsort(suma_capacidades)[::-1]
            k = k_2[0]  # Solucionamos aquella capacidad que sea mas grande
            while True:
                SD_mismos_recursos = [i for i, v in enumerate(lista_clases_SD) if np.array_equal(v, lista_clases_SD[k]) and i != k]  # Sacamos índices de SD con mismos recurso
                k_3 = random.choice(SD_mismos_recursos)  # Jugamos con uno de los 4 SD con menos suma de bases
                indices_bases_SD = [j for j, value in enumerate(individuo) if value == k]  # Obtenemos índices de las bases cuya suma de caps supera el umbral
                indices_resto_bases = [j for j, value in enumerate(individuo) if value == k_3]  # Obtenemos índices del resto de bases
                capacidades_bases_SD_ordenados = list(np.argsort([capacidades[i] for i in indices_bases_SD])[::-1])
                indices_bases_SD_ordenados = [indices_bases_SD[i] for i in capacidades_bases_SD_ordenados]

                if (suma_capacidades[k] > 200 and suma_capacidades[k] < 220) or (suma_capacidades[k] < 200 and suma_capacidades[k] > 180):
                    indice_base_1 = indices_bases_SD_ordenados[len(indices_bases_SD_ordenados) - np.random.randint(1,len(indices_bases_SD_ordenados))]  # Cuando se estabilice la suma de capacidades cogemos caps pequeñas
                    lista_filtrada = [value for value in indices_resto_bases if capacidades[value] <= capacidades[indice_base_1]]
                    if lista_filtrada:
                        indice_base_aleatoria_2 = random.choice(lista_filtrada)  # Elección aleatoria de la base del resto de bases
                    else:
                        if indices_resto_bases:
                            indice_base_aleatoria_2 = random.choice(indices_resto_bases)
                        else:
                            SD_mismos_recursos_2 = [v for i, v in enumerate(SD_mismos_recursos) if v != k and v != k_3]
                            lista_ind = []
                            while True: #Cotejamos que haya bases de la misma clase en otros SD
                                k_4 = random.choice(SD_mismos_recursos_2)  # Jugamos con uno de los SD con mismos recursos que las bases
                                indices_resto_bases = [j for j, value in enumerate(individuo) if value == k_4 and value not in lista_ind]
                                if indices_resto_bases:
                                    break
                                else:   #Si no hay, la añadimos a una lista
                                    lista_ind.append(k_4)
                                    if len(lista_ind) == len(SD_mismos_recursos_2): #Cuando el tamaño de lista sea igual que SD_mismos_recursos...
                                        e = random.randint(0, 5)
                                        f = indices_bases_SD_ordenados[0:e]
                                        individuo[f] = k_3  #... Descargamos algunas bases del SD que nos da problemas sobre el otro (k_3)
                                    else:
                                        continue
                            indice_base_aleatoria_2 = random.choice(indices_resto_bases)

                else:
                    indice_base_1 = indices_bases_SD_ordenados[0]
                    #indice_base_1 = random.choice(indices_bases_SD_ordenados[0:3])  # Elegimos una de las 5 bases del SD con mayor capacidad
                    lista_filtrada = [value for value in indices_resto_bases if capacidades[value] < capacidades[indice_base_1]]
                    if lista_filtrada:
                        indice_base_aleatoria_2 = random.choice(lista_filtrada)  # Elección aleatoria de la base del resto de bases
                    else:
                        if indices_resto_bases:
                            indice_base_aleatoria_2 = random.choice(indices_resto_bases)
                        else:
                            SD_mismos_recursos_2 = [v for i, v in enumerate(SD_mismos_recursos) if v != k and v != k_3]
                            lista_ind = []
                            while True:  # Cotejamos que haya bases de la misma clase en otros SD
                                k_4 = random.choice(SD_mismos_recursos_2)  # Jugamos con uno de los SD con mismos recursos que las bases
                                indices_resto_bases = [j for j, value in enumerate(individuo) if value == k_4 and value not in lista_ind]
                                if indices_resto_bases:
                                    break
                                else:  # Si no hay, la añadimos a una lista
                                    lista_ind.append(k_4)
                                    if len(lista_ind) == len(SD_mismos_recursos_2):  # Cuando el tamaño de lista sea igual que SD_mismos_recursos...
                                        e = random.randint(0, 5)
                                        f = indices_bases_SD_ordenados[0:e]
                                        individuo[f] = k_3  # ... Descargamos algunas bases del SD que nos da problemas sobre el otro (k_3)
                                    else:
                                        continue
                            indice_base_aleatoria_2 = random.choice(indices_resto_bases)

                if abs(200 - suma_capacidades[k_3]) < 20 and suma_capacidades[k_3] < 200:
                    individuo[indice_base_1], individuo[indice_base_aleatoria_2] = individuo[indice_base_aleatoria_2], individuo[indice_base_1]  # Intercambio posiciones de las bases
                else:
                    e = random.randint(0, 5)
                    f = indices_bases_SD_ordenados[0:e]
                    individuo[f] = k_3
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

if __name__ == "__main__":

    Pob_Actual = []
    Costes = []
    poblacion_inicial = 100
    Num_Gen = 500
    numero_bases = 200
    numero_supply_depots = 15
    capacidad_maxima = 20
    numero_clases = 5
    puntos = list(Puntos_Sin_Repetir(numero_bases+numero_supply_depots))
    supply_depots = puntos[-numero_supply_depots:]
    bases = puntos[:numero_bases]
    longitudes_bases, latitudes_bases = zip(*bases)
    capacidad_bases = np.random.randint(1, capacidad_maxima, size=(numero_bases))
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    longitudes_supply_depots, latitudes_supply_depots = zip(*supply_depots)
    capacidad_supply_depots = np.full(numero_supply_depots,200)

    lista_clases_base = []
    for i in range(numero_bases):
        vector_clase = np.zeros(numero_clases, dtype=int)
        indice_clase = random.choice([i for i, v in enumerate(vector_clase)])
        vector_clase[indice_clase] = 1
        lista_clases_base.append(vector_clase)

    lista_clases_SD = []
    indices_SD_validos = list(np.arange(numero_supply_depots))  # Sacamos lista de índices de cada SD
    indice_seleccionado = []
    for i in range(numero_supply_depots):
        indice_SD = random.sample([j for j in indices_SD_validos if j not in indice_seleccionado], 1)
        indice_seleccionado.extend(indice_SD)
        vector_clase = np.zeros(numero_clases, dtype=int)
        indice_clase = indice_SD[
                           0] % numero_clases  # Lo que hacemos es asegurarnos que todos los SD tienen una clase distinta
        vector_clase[indice_clase] = 1
        lista_clases_SD.append(vector_clase)

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
        "Ngen": Num_Gen,  #Número de generaciones
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
    colores = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'yellow', 'magenta', 'cyan', 'violet','lime', 'gold', 'silver', 'indigo']
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p',label='Puntos de Suministro')
    for k in range(numero_supply_depots):
        SD = np.array([i for i, v in enumerate(solution) if v == k])
        color = colores[k]  # Un color por cada iteración
        for j in range(len(SD)):
            plt.scatter(longitudes_bases[SD[j]], latitudes_bases[SD[j]], color=color, label='Bases')
        plt.plot([longitudes_bases[SD[0]], longitudes_supply_depots[k]],[latitudes_bases[SD[0]], latitudes_supply_depots[k]], color='red')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()