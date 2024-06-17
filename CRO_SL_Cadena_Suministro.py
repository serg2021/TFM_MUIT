import numpy as np
from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
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

    def random_solution(self):  #Generamos una población inicial -> Solo indicamos cómo serán las soluciones de la población y las reparamos una vez se generen, el resto lo hace el algoritmo
        Pob_Ini = np.random.randint(0, numero_supply_depots, size=(1, self.size))  # Solución tipo -> 2 FILAS: 1 solución y otra para lista de bases a inters
        x = np.full(self.size, 200)
        Pob_Ini = np.vstack((Pob_Ini, x))
        if(Comprobacion_Individuo(Pob_Ini, capacidad_bases, distancias_euclideas)):
            Pob_Ini = Reparacion_Mayor_Menor(Pob_Ini, capacidad_bases, distancias_euclideas)
        return Pob_Ini

    def repair_solution(self, solution):    #Reparación de individuos
        for i in range(numero_bases):
            if solution[0][i] > 9 or solution[0][i] < 0:
                solution[0][i] = np.random.randint(0, numero_supply_depots)
        if (Comprobacion_Individuo(solution, capacidad_bases, distancias_euclideas)):
            solution = Reparacion_Mayor_Menor(solution, capacidad_bases, distancias_euclideas)
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
    if isinstance(base, list) and isinstance(supply, list): #Cálculo de todas las distancias de bases e inters a SDs
        x_supply, y_supply = zip(*supply)
        x_base, y_base = zip(*base)
        dist = []
        for i in range(len(supply)):
            dist_aux = []
            for j in range(len(base)):
                distancia = math.sqrt((x_base[j] - x_supply[i]) ** 2 + (y_base[j] - y_supply[i]) ** 2)
                dist_aux.append(distancia)
            dist.append(dist_aux)
    else:   #Cálculo de distancia de una base al inter
        x_supply, y_supply = supply
        x_base, y_base = base
        dist = math.sqrt((x_base - x_supply) ** 2 + (y_base - y_supply) ** 2)
    return dist
def Funcion_Fitness(distancias, individuo):
    fitness = 0
    for j in range(len(individuo[0])):   #Bucle para recorrer bases que no sean intermediarios
        SD_base = individuo[0][j]    #Saco el SD asociado a una base de la población
        if(SD_base > 9 or SD_base < 0 or isinstance(SD_base, float)):   #Está mutando y nos da valores de SD que no pueden ser -> SOLUCIÓN:
            SD_base = np.random.randint(0,numero_supply_depots)                                   # Se genera el número a modificar
        ind_inter = np.where(individuo[0][ind_intermediarios] == SD_base)[0]   #Buscamos qué intermediarios tienen ese SD asociado
        for k in ind_inter: #Comprobamos para esos intermediarios sus distancias con la base elegida
            distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[k])
            if distancia_base_inter < distancias[SD_base][j]:   #Si esa distancia es menor que la de la base al SD
                fitness += distancia_base_inter  # Calculo fitness usando la distancia de la base al intermediario
            else:
                continue
        fitness += distancias[SD_base][j]    #Calculo fitness buscando en la matriz de distancias la distancia asociada
    fitness = fitness/numero_bases
    return fitness

def Comprobacion_Individuo (individuo, capacidades, distancias):
    suma_comprobar = list(np.zeros(numero_supply_depots))
    for i in range(numero_supply_depots):
        comprobar_capacidades = list(np.zeros(numero_bases))  # Array auxiliar del tamaño del número de bases para almacenar las capacidades
        indices_bases = np.where(individuo[0][ind_bases_antes] == i)[0]  # Obtenemos los indices de las bases asociadas a un SD "i"
        for ind in ind_bases_antes[indices_bases]:
            comprobar_capacidades[ind] = capacidades[ind]  # Copiamos las capacidades originales en otra variable auxiliar
        ind_inter = np.where(individuo[0][ind_intermediarios] == i)[0]  # Buscamos qué intermediarios tienen ese SD asociado
        for ind in ind_intermediarios[ind_inter]:
            comprobar_capacidades[ind] = capacidades[ind]  # Copiamos las capacidades originales en otra variable auxiliar

        # YA TENEMOS LAS CAPACIDADES COPIADAS -> PROCEDEMOS A LA COMPROBACIÓN

        for k in ind_intermediarios[ind_inter]:  # Comprobamos para esos intermediarios sus distancias con el SD y con las bases cercanas
            contador = 0  # Con este contador vamos sumando las capacidades de las bases que coteja el intermediario
            indices_bases_inter = list(np.full(numero_bases, numero_bases))  # Lista de índices de bases a intermediarios
            for j in ind_bases_antes[indices_bases]:
                distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[k])  # Distancia entre una base y el intermediario
                if distancia_base_inter < distancias[i][j] and capacidades[j] <= (comprobar_capacidades[k] - contador):
                    # Si esa distancia es menor que la de la base al SD
                    # y la capacidad de la base es menor o igual que la diferencia entre la cap del intermediario y la suma de cap de las bases que contempla...
                    comprobar_capacidades[j] = 0  # No contemplamos la capacidad de esa base -> Va englobada en la capacidad del intermediario
                    contador += capacidades[j]  # Sumamos esa capacidad al contador y vemos la siguiente base
                    indices_bases_inter[j] = k
                    if indices_bases_inter[j] in ind_intermediarios:
                        individuo[1][j] = indices_bases_inter[j]  # Añadimos en el indice que corresponde con el número del SD los índices recogidos

        individuo = individuo.astype(int)
        A = individuo[1][np.array([i for i, x in enumerate(individuo[1]) if x != 200], dtype=int)]
        B = individuo[0][A]
        C = individuo[0][np.array([i for i, x in enumerate(individuo[1]) if x != 200], dtype=int)]
        suma_comprobar[i] = sum(comprobar_capacidades)
        if not np.array_equal(B,C):  # Si la suma de las capacidades de una solucion para un supply depot es mayor que 200 -> REPARACION
            return True
    Caps_Comprobar = [ind_cap for ind_cap, j in enumerate(suma_comprobar) if j > 200]
    if len(Caps_Comprobar) > 0:
        return True

def Reparacion_Mayor_Menor (individuo, capacidades, distancias): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
    individuo = individuo.astype(int)
    capacidades_sd = list(np.zeros(numero_supply_depots))  # Capacidades de los SD
    suma_capacidades = list(np.zeros(numero_supply_depots))  # Suma de las capacidades de las bases
    individuo[1, :] = numero_bases  # Limpiamos la fila de intermediarios
    for i in range(numero_supply_depots):
        capacidades_sd_i = list(np.zeros(numero_bases))  # Array auxiliar del tamaño del número de bases para almacenar las capacidades
        indices_bases = np.where(individuo[0][ind_bases_antes] == i)[0]  #Obtenemos los indices de las bases asociadas a un SD "i"
        for ind in ind_bases_antes[indices_bases]:
            capacidades_sd_i[ind] = capacidades[ind]   #Copiamos las capacidades originales en otra variable auxiliar
        ind_inter = np.where(individuo[0][ind_intermediarios] == i)[0]  # Buscamos qué intermediarios tienen ese SD asociado
        for ind in ind_intermediarios[ind_inter]:
            capacidades_sd_i[ind] = capacidades[ind]   #Copiamos las capacidades originales en otra variable auxiliar
        capacidades_sd[i] = capacidades_sd_i

        # CAPACIDADES COPIADAS

        for k in ind_intermediarios[ind_inter]:  # Comprobamos para esos intermediarios sus distancias con el SD y con las bases cercanas
            contador = 0  # Con este contador vamos sumando las capacidades de las bases que coteja el intermediario
            indices_bases_inter = list(np.full(numero_bases, numero_bases))  # Lista de índices de bases a intermediarios
            for j in ind_bases_antes[indices_bases]:
                distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[k])  # Distancia entre una base y el intermediario
                if distancia_base_inter < distancias[i][j] and capacidades[j] <= (capacidades[k] - contador):
                    # Si esa distancia es menor que la de la base al SD
                    # y la capacidad de la base es menor o igual que la diferencia entre la cap del intermediario y la suma de cap de las bases que contempla...
                    capacidades_sd_i[j] = 0  # No contemplamos la capacidad de esa base -> Va englobada en la capacidad del intermediario
                    contador += capacidades[j]  # Sumamos esa capacidad al contador y vemos la siguiente base
                    indices_bases_inter[j] = k  #Guardamos el valor del intermediario en la posición de la base
                if indices_bases_inter[j] in ind_intermediarios:
                    individuo[1][j] = indices_bases_inter[j]   #Añadimos en el indice que corresponde con el número del SD los índices recogidos
        #for s in range(numero_bases):
            #if individuo[1][s] not in ind_intermediarios:
                #individuo[1][s] = numero_bases  #Nos aseguramos de que el valor que no esté dentro de ind_intermediarios no interfiera
        suma_capacidades[i] = sum(capacidades_sd_i)  # Almacenamos todas las sumas de las capacidades en un array

    # YA TENEMOS LAS CAPACIDADES COPIADAS -> PROCEDEMOS A LA COMPROBACIÓN

    Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  #Comprobamos en qué SD's se han superado la capacidad
    if len(Caps_SD_Superadas) > 0:
        while True:
            k_2 = np.argsort(suma_capacidades)[::-1]
            k = k_2[0]  #Solucionamos aquella capacidad que sea mas grande
            while True:
                k_3 = random.choice(k_2[len(suma_capacidades) - 3:len(suma_capacidades)])
                indices_bases_inters_SD = [j for j, value in enumerate(individuo[0]) if value == k]    #Obtenemos índices de las bases cuya suma de caps supera el umbral
                indices_resto_bases = [j for j, value in enumerate(individuo[0]) if value == k_3]  # Obtenemos índices del resto de bases
                capacidades_bases_SD_ordenados = list(np.argsort([capacidades[i] for i in indices_bases_inters_SD])[::-1])
                indices_bases_SD_ordenados = [indices_bases_inters_SD[i] for i in capacidades_bases_SD_ordenados]

                indice_base_1 = indices_bases_SD_ordenados[0] #Elegimos una de las 5 bases del SD con mayor capacidad
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
                if 200 - suma_capacidades[k_2[9]] < 50:
                    individuo[0][indice_base_1], individuo[0][indice_base_aleatoria_2] = individuo[0][indice_base_aleatoria_2], individuo[0][indice_base_1] #Intercambio posiciones de las bases
                else:
                    e = random.randint(0,5)
                    f = indices_bases_SD_ordenados[0:e]
                    individuo[0][f] = k_2[9]
                indices_bases = np.where(individuo[0][ind_bases_antes] == k)[0]  # Obtenemos los indices de las bases asociadas a un SD "i"
                capacidades_sd_i = list(np.zeros(numero_bases))  # Array auxiliar del tamaño del número de bases para almacenar las capacidades
                for ind in ind_bases_antes[indices_bases]:
                    capacidades_sd_i[ind] = capacidades[ind]  # Copiamos las capacidades originales en otra variable auxiliar
                ind_inter = np.where(individuo[0][ind_intermediarios] == k)[0]  # Buscamos qué intermediarios tienen ese SD asociado
                for ind in ind_intermediarios[ind_inter]:
                    capacidades_sd_i[ind] = capacidades[ind]  # Copiamos las capacidades originales en otra variable auxiliar
                capacidades_sd[k] = capacidades_sd_i

                # TENEMOS CAPACIDADES ACTUALIZADAS

                if indice_base_1 in ind_intermediarios:
                    indices_bases_inters = [j for j, value in enumerate(individuo[1]) if value == indice_base_1]
                    individuo[1][indices_bases_inters] = 200  # Debido a que podemos arrastrar valores erróneos, después de reparar limpiamos la lista de índices y luego la formamos
                    for i in indices_bases_inters:
                        capacidades_sd_i[i] = capacidades[i]
                if indice_base_aleatoria_2 in ind_intermediarios:
                    indices_bases_inters = [j for j, value in enumerate(individuo[1]) if value == indice_base_aleatoria_2]
                    individuo[1][indices_bases_inters] = 200  # Debido a que podemos arrastrar valores erróneos, después de reparar limpiamos la lista de índices y luego la formamos
                    for i in indices_bases_inters:
                        capacidades_sd_i[i] = capacidades[i]
                individuo[1][indice_base_1] = 200
                capacidades_sd_i[indice_base_1] = capacidades[indice_base_1]
                individuo[1][indice_base_aleatoria_2] = 200
                capacidades_sd_i[indice_base_aleatoria_2] = capacidades[indice_base_aleatoria_2]

                #indices_bases_reparadas = [j for j, value in enumerate(individuo) if value == k]    #Obtenemos índices de las bases cuya suma de caps supera el umbral

                for v in ind_intermediarios[ind_inter]:  # Comprobamos para esos intermediarios sus distancias con el SD y con las bases cercanas
                    contador = 0  # Con este contador vamos sumando las capacidades de las bases que coteja el intermediario
                    indices_bases_inter = list(np.full(numero_bases, 200))  # Lista de índices de bases a intermediarios
                    for j in ind_bases_antes[indices_bases]:
                        distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[v])  # Distancia entre una base y el intermediario
                        if distancia_base_inter < distancias[k][j] and capacidades[j] <= (capacidades[v] - contador):
                            # Si esa distancia es menor que la de la base al SD
                            # y la capacidad de la base es menor o igual que la diferencia entre la cap del intermediario y la suma de cap de las bases que contempla...
                            capacidades_sd_i[j] = 0  # No contemplamos la capacidad de esa base -> Va englobada en la capacidad del intermediario
                            contador += capacidades[j]  # Sumamos esa capacidad al contador y vemos la siguiente base
                            indices_bases_inter[j] = v  # Guardamos el valor del intermediario en la posición de la base
                            individuo[1][j] = indices_bases_inter[j]  # Añadimos en el indice que corresponde con el número del SD los índices recogidos
                for s in ind_intermediarios[ind_inter]:
                    counter = 0
                    for t in range(len(individuo[1])):
                        if individuo[1][t] == s:    #Si una base pertenece a un intermediario añadimos su capacidad
                            counter += capacidades[t]
                    if capacidades[s] < counter:    #Si la suma de caps de bases supera al intermediario, salimos de los bucles y se vuelve a hacer el while True
                        indices_bases_mas = np.array([i for i, value in enumerate(individuo[1]) if value == s])  #Sacamos índices de bases asociadas al intermediario
                        capacidades_orden = list(np.argsort([capacidades[i] for i in indices_bases_mas])[::-1])
                        individuo[1][indices_bases_mas[capacidades_orden[0]]] = 200    #La base que tenga más capacidad deja de pertenecer a ese intermediario -> Priorizamos las pequeñas
                        capacidades_sd_i[indices_bases_mas[capacidades_orden[0]]] = capacidades[indices_bases_mas[capacidades_orden[0]]]

                F = np.array([i for i, x in enumerate(individuo[0]) if x == k], dtype=int)
                H = np.where(individuo[1][F] == 200)
                I = np.array(capacidades)
                J = np.array(capacidades_sd_i)
                if sum(I[F[H]]) > 200: #Si es más de 200, volvemos a hacer mismo código del while
                    continue
                elif sum(I[F[H]]) == sum(J[F[H]]): #Si no, salimos del while y avanzamos en el for
                    capacidades_sd[k] = capacidades_sd_i
                    suma_capacidades[k] = sum(I[F[H]])
                    break

            #HAY QUE MIRAR DE NUEVO LO DE LAS CAPACIDADES -> LO MEJOR SERÁ VOLVER A ANALIZARLO PASO A PASO CON CROQUIS (AL MENOS SOLO ESTA PARTE)
            #LA COMPROBACIÓN DE INDIVIDUO PARECE ESTAR BIEN, Y LA ASIGNACIÓN DE INTERMEDIARIOS SEGÚN CAPACIDADES Y QUE TODAS LAS BASES E INTERS VAYAN AL MISMO SD TAMBIÉN
            #REPITO -> SÓLO QUEDA SOLUCIONAR CAPACIDADES DE LOS SD!!!

            #CUANDO SOLUCIONAMOS UNA SUMA, LA GUARDAMOS, Y SOLUCIONAMOS LA SIGUIENTE SIN ASEGURARNOS DE QUE LA ANTERIOR ESTÉ BIEN
            #TENEMOS QUE AÑADIR CÓDIGO PARA VER QUE NO PASE ESO Y QUE TODAS LAS SUMAS ESTÉN BIEN ACTUALIZADAS

            for i in range(numero_supply_depots):
                F = np.array([t for t, x in enumerate(individuo[0]) if x == i], dtype=int)
                H = np.where(individuo[1][F] == 200)
                I = np.array(capacidades)
                suma_capacidades[i] = sum(I[F[H]])

            Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad
            if len(Caps_SD_Superadas) == 0:
                break
    return individuo

if __name__ == "__main__":

    Pob_Actual = []
    Lista_Bases_Actual = []
    Costes = []
    poblacion_inicial = 100
    numero_bases = 200
    numero_supply_depots = 10
    capacidad_maxima = 20
    capacidad_bases = np.zeros(numero_bases)
    puntos = list(Puntos_Sin_Repetir(numero_bases+numero_supply_depots))
    supply_depots = puntos[-numero_supply_depots:]
    bases = puntos[:numero_bases]
    ind_intermediarios = np.array(random.sample(range(len(bases)),int(len(bases)*0.2)))   #Extraigo índices de intermediarios de forma aleatoria
    intermediarios = [bases[i] for i in ind_intermediarios]
    ind_bases_antes = np.array([i for i, elemento in enumerate(bases) if elemento not in intermediarios])
    bases = list(set(bases) - set(intermediarios))  #Actualizamos el número de bases sin contar intermediarios
    longitudes_bases, latitudes_bases = zip(*bases)
    longitudes_inter, latitudes_inter = zip(*intermediarios)
    capacidad_bases[ind_bases_antes] = np.random.randint(1, capacidad_maxima, size=len(bases))
    capacidad_bases[ind_intermediarios] = np.random.randint(10, capacidad_maxima, size=len(intermediarios))
    capacidad_bases = list(capacidad_bases)
    #capacidad_inters = np.random.randint(10, capacidad_maxima, size=(len(intermediarios)))  #Les doy capacidades
    #capacidad_bases = np.concatenate((capacidad_bases, capacidad_inters))
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    bases_inter = puntos[:numero_bases] #Recuperamos datos de bases e intermediarios
    longitudes_supply_depots, latitudes_supply_depots = zip(*supply_depots)
    capacidad_supply_depots = np.full(numero_supply_depots,200)

    distancias_euclideas = Distancia_Base_Supply_Depot_2D(bases_inter, supply_depots) #Obtenemos distancias de bases a supply depots

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
        "Ngen": 120,  #Número de generaciones
        "Neval": 3e3,   #Número de evaluaciones de la función objetivo
        "fit_target": 50,   #Valor de función objetivo a alcanzar -> Ponemos 50 por poner un valor muy bajo

        "verbose": True,    #Informe periódico de cómo va el algoritmo
        "v_timer": 1,   #Tiempo entre informes generados
        "Njobs": 1, #Número de trabajos a ejecutar en paralelo -> Como es 1, se ejecuta de forma secuencial

        "dynamic": True,    #Determina si usar la variante dinámica del algoritmo -> Permite cambiar el tamaño de cada substrato (Mirar paper)
        "dyn_method": "success",    #Determina la probabilidad de elegir un substrato para cada coral en la siguiente generación -> Con 'success' usa el ratio de larvas exitosas en cada generación
        "dyn_metric": "best",    #Determina cómo agregar los valores de cada substrato para obtener la medida de cada uno
        "dyn_steps": 10,    #Número de evaluaciones por cada substrato
        "prob_amp": 0.01    #Determina cómo las diferencias entre las métricas de los substratos afectan la probabilidad de cada una -> Cuanto más pequeña, más amplifica
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
        print("Base " + str(j) + "-> SD: " + str(solution[0][j]) + " -> Intermediario: " + str(solution[1][j]) + " -> Capacidad: " + str(capacidad_bases[j]))
    print("Coste final: " + str(obj_value))

    #Graficamos la solución
    lista_base_indices = []
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_inter, latitudes_inter, color='green', label='Intermediarios')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    bases = puntos[:numero_bases]
    longitudes_bases, latitudes_bases = zip(*bases)
    for v in range(len(ind_intermediarios)):
        if isinstance(solution[1], float):
            solution[1] = list(np.zeros(numero_bases))
        base_indices = [i for i, x in enumerate(solution[1]) if x == ind_intermediarios[v]]
        for j in base_indices:
            plt.plot([longitudes_bases[j], longitudes_inter[v]],[latitudes_bases[j], latitudes_inter[v]], color='yellow')
        lista_base_indices.extend(base_indices)
    for k in range(numero_bases):
        solution[0][k] = int(solution[0][k])
        if k not in lista_base_indices:
            plt.plot([longitudes_bases[k],longitudes_supply_depots[solution[0][k]]], [latitudes_bases[k], latitudes_supply_depots[solution[0][k]]],color='red')
        if k in ind_intermediarios:
            plt.plot([longitudes_bases[k],longitudes_supply_depots[solution[0][k]]], [latitudes_bases[k], latitudes_supply_depots[solution[0][k]]],color='red')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()