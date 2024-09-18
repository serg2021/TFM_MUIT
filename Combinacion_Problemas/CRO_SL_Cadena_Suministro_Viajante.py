import numpy as np
from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateInt import *
import random
import math
import matplotlib.pyplot as plt
import os
import csv


class Fitness(AbsObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size    #Número de bases
        super().__init__(self.size, opt)

    def objective(self, solution):  #Función objetivo -> El algoritmo se encarga de hacerlo, no es como el evolutivo
        if len(solution) == numero_bases:   #Fitness para una solución genérica
            return Funcion_Fitness(distancias_euclideas, solution)
        if len(solution) < numero_bases:
            if len(solution) == len(indices_bases_inter) and flag_1 == True:  #Fitness para viajante en intermediarios
                return Funcion_Fitness_Viajante_Intermediario(dist_bases_list_inter, solution)
            elif len(solution) == len(indices_bases_inter) and flag_2 == True:   #Fitness para viajante normal
                solution_normal[0].astype(int)
                solution_normal[1].astype(int)
                return Funcion_Fitness_Viajante(dist_bases_list_SD, distancias_euclideas, solution, solution_normal,indices_bases_SD[indices_bases_inter])


    def random_solution(self):  #Generamos una población inicial -> Solo indicamos cómo serán las soluciones de la población y las reparamos una vez se generen, el resto lo hace el algoritmo
        Pob_Ini = np.random.randint(0, len(indices_bases_inter), size=self.size)  # Solución tipo
        if(Comprobacion_Individuo_Viajante(Pob_Ini)):
            Pob_Ini = Reparacion_Viajante(Pob_Ini)
        return Pob_Ini

    def repair_solution(self, solution):    #Reparación de individuos
        if len(solution) == numero_bases:
            for i in range(numero_bases):
                if solution[i] > numero_supply_depots-1 or solution[i] < 0:
                    solution[i] = np.random.randint(0, numero_supply_depots)
            if (Comprobacion_Individuo(solution, capacidad_bases, distancias_euclideas)):
                solution = Reparacion_Mayor_Menor(solution, capacidad_bases, distancias_euclideas)
                #Lo hemos reparado en base a la capacidad -> PERO NO EN BASE A LOS SD POSIBLES -> SALEN VALORES DE SD QUE NO SON
                #Tenemos que añadir AQUÍ una forma de repararlo -> Lo más sencillo es hacer un bucle que recorra cada elemento de la solución y
                #Cambiar esos valores por otros aleatorios que estén dentro del rango
        elif len(solution) < numero_bases:
            for i in range(len(solution)):  #Reparamos la mutación si usa valores que excedan el límite de tamaño del coral
                if solution[i] > len(solution)-1 or solution[i] < 0:
                    solution[i] = np.random.randint(0, len(solution))
            if (Comprobacion_Individuo_Viajante(solution)):
                solution = Reparacion_Viajante(solution)
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
    elif isinstance(base, list) and isinstance(supply, tuple):  # Cálculo de todas las distancias de bases a otra base
        x_supply, y_supply = supply
        x_base, y_base = zip(*base)
        dist = []
        for j in range(len(base)):
            distancia = math.sqrt((x_base[j] - x_supply) ** 2 + (y_base[j] - y_supply) ** 2)
            dist.append(distancia)
    else:  # Cálculo de distancia de una base al inter
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

def Funcion_Fitness_Viajante(distancias, dist, individuo, pob, indices):
    SD = pob[0][indices[0]]
    fitness = 0
    indices_orden = list(np.argsort(individuo)) #Sacamos el orden de los índices para verlos de forma consecutiva
    for j in range(len(indices_orden)-1):
        k = j +1
        fitness += distancias[indices_orden[j]][indices_orden[k]]    #Calculo fitness buscando en la matriz de distancias la distancia asociada
    fitness += dist[SD][indices[indices_orden[0]]]
    fitness += dist[SD][indices[indices_orden[len(indices_orden)-1]]]   #Sumamos distancia del camino de vuelta
    fitness = fitness/(len(individuo)+1)
    return fitness

def Funcion_Fitness_Viajante_Intermediario(distancias, individuo):
    fitness = 0
    indices_orden = list(np.argsort(individuo))  #Sacamos el orden de los índices para verlos de forma consecutiva
    for j in range(len(indices_orden)-1):
        k = j+1
        fitness += distancias[indices_orden[j]][indices_orden[k]]    #Calculo fitness buscando en la matriz de distancias la distancia asociada
    fitness += distancias[indices_orden[0]][len(indices_orden)]         #Sumamos distancia desde inter hasta base
    fitness += distancias[indices_orden[len(indices_orden) - 1]][len(indices_orden)]  # Sumamos distancia de vuelta al inter
    fitness = fitness/(len(individuo)+1)# -> Aquí normalizaríamos, pero perderíamos información de distancias totales en los intermediarios para la solución general
    return fitness

def Comprobacion_Individuo_Viajante (individuo):
    set_individuo = set(individuo)  #Lo pasamos a set, ya que en un set no hya elementos duplicados
    if len(individuo) != len(set_individuo):    #Comprobamos longitudes -> Si son distintas, quiere decir que había duplicados
        return True
def Reparacion_Viajante(individuo):   #Lo que haremos será asegurarnos de que no se repiten números en toda la lista
    valores_posibles = set([i for i, v in enumerate(individuo)])
    valores_sin_repetir = set(individuo)
    valores_validos = valores_posibles - valores_sin_repetir
    for i in range(len(individuo)-1):
        if len(valores_validos) == 0:
            break
        for j in range(i+1,len(individuo)):
            if individuo[i] == individuo[j]:
                indices = [i, j]
                indice_reparado = random.choice(indices)    #Elegimos uno de los índices a reparar aleatoriamente
                individuo[indice_reparado] = random.choice(list(valores_validos))   #Elegimos un valor aleatorio de los posibles
                valores_validos.remove(individuo[indice_reparado]) #Eliminamos el nuevo valor de los valores válidos
            if len(valores_validos) == 0:
                break
    return individuo

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
        A = individuo[1][np.array([i for i, x in enumerate(individuo[1]) if x != 200], dtype=int)]  #Saco índices de bases asociadas a inters
        B = individuo[0][A]     #Miro el SD al que están asociadas
        C = individuo[0][np.array([i for i, x in enumerate(individuo[1]) if x != 200], dtype=int)]  #Saco los SD's asociados a los inters
        suma_comprobar[i] = sum(comprobar_capacidades)
        if not np.array_equal(B,C):  #Si B y C no son iguales -> Reparación -> Porque no puede haber bases asociadas a un inter que vayan a un SD distinto al del inter
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
                k_3 = random.choice(k_2[len(suma_capacidades) - 4:len(suma_capacidades)])   #Jugamos con uno de los 4 SD con menos suma de bases
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
                if abs(200 - suma_capacidades[k_2[9]]) < 50 and suma_capacidades[k_2[9]] < 200:
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
                #J = np.array(capacidades_sd_i)
                if sum(I[F[H]]) > 200: #Si es más de 200, volvemos a hacer mismo código del while
                    continue
                else: #Si no, salimos del while y avanzamos en el for
                    capacidades_sd[k] = capacidades_sd_i
                    suma_capacidades[k] = sum(I[F[H]])
                    break

            #CUANDO SOLUCIONAMOS UNA SUMA, LA GUARDAMOS, Y SOLUCIONAMOS LA SIGUIENTE SIN ASEGURARNOS DE QUE LA ANTERIOR ESTÉ BIEN
            #TENEMOS QUE AÑADIR CÓDIGO PARA VER QUE NO PASE ESO Y QUE TODAS LAS SUMAS ESTÉN BIEN ACTUALIZADAS

            for i in range(numero_supply_depots):   #Bucle para comprobar sumas de capacidades alrededor de un SD
                F = np.array([t for t, x in enumerate(individuo[0]) if x == i], dtype=int)
                H = np.where(individuo[1][F] == 200)
                I = np.array(capacidades)
                suma_capacidades[i] = sum(I[F[H]])

            Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad
            if len(Caps_SD_Superadas) == 0:
                break
    return individuo

if __name__ == "__main__":

    random.seed(2039)
    np.random.seed(2039)
    Pob_Actual = []
    Lista_Bases_Actual = []
    Costes = []
    poblacion_inicial = 200
    numero_bases = 200
    numero_supply_depots = 10
    capacidad_maxima = 18
    Ruta_Puntos = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Orografia',
        f"Bases_SD_2.csv")
    Ruta_Intermediarios = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Cadena_Suministro',
        f"Intermediarios_2.csv")
    Ruta_Capacidades = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Cadena_Suministro',
        f"Cap_Bases_SD_2.csv")
    Ruta_Solucion = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Combinacion_Problemas',
        f"Solucion_2.csv")
    if not os.path.exists(Ruta_Puntos):
        puntos = list(Puntos_Sin_Repetir(numero_bases + numero_supply_depots))
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
    capacidad_bases = np.zeros(numero_bases)
    supply_depots = puntos[-numero_supply_depots:]
    bases = puntos[:numero_bases]
    if not os.path.exists(Ruta_Intermediarios):
        ind_intermediarios = np.array(random.sample(range(len(bases)), int(len(bases) * 0.2)))  # Extraigo índices de intermediarios de forma aleatoria
        np.savetxt(Ruta_Intermediarios, ind_intermediarios, delimiter=',')
    else:
        ind_intermediarios = []
        with open(Ruta_Intermediarios, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                ind_intermediarios.append(int(numbers[0]))
            ind_intermediarios = np.array(ind_intermediarios)
    intermediarios = [bases[i] for i in ind_intermediarios]
    ind_bases_antes = np.array([i for i, elemento in enumerate(bases) if elemento not in intermediarios])
    bases = list(set(bases) - set(intermediarios))  #Actualizamos el número de bases sin contar intermediarios
    latitudes_bases, longitudes_bases = zip(*bases)
    latitudes_inter, longitudes_inter = zip(*intermediarios)
    if not os.path.exists(Ruta_Capacidades):
        capacidad_bases[ind_bases_antes] = np.random.randint(1, capacidad_maxima, size=len(bases))
        capacidad_bases[ind_intermediarios] = np.random.randint(10, capacidad_maxima, size=len(intermediarios))
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
    capacidad_bases = list(capacidad_bases)
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    bases_inter = puntos[:numero_bases] #Recuperamos datos de bases e intermediarios
    latitudes_supply_depots, longitudes_supply_depots = zip(*supply_depots)
    capacidad_supply_depots = np.full(numero_supply_depots,200)

    distancias_euclideas = Distancia_Base_Supply_Depot_2D(bases_inter, supply_depots) #Obtenemos distancias de bases a supply depots

    ## Hasta aquí generamos las bases y SD que vamos a tener antes del algoritmo -> Junto con las distancias asociadas

    params_Inter = {                      #Hiperparámetros del algoritmo
        "popSize": poblacion_inicial, #Población inicial
        "rho": 0.6, #Porcentaje de ocupación de corales del Reef inicial
        "Fb": 0.98, #Proporción de Broadcast Spawning
        "Fd": 0.5,  #Proporción de Depredación
        "Pd": 0.8,  #Probabilidad de Depredación
        "k": 3, #Número máximo de intentos para que la larva intente asentarse
        "K": 20,    #Número máximo de corales con soluciones duplicadas
        "group_subs": True, #Si 'True', los corales se reproducen sólo en su mismo substrato, si 'False', se reproducen con toda la población

        "stop_cond": "Neval",   #Condición de parada
        "time_limit": 4000.0,   #Tiempo límite (real, no de CPU) de ejecución
        "Ngen": 100,  #Número de generaciones
        "Neval": 15050,   #Número de evaluaciones de la función objetivo
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

    params = {  # Hiperparámetros del algoritmo
        "popSize": poblacion_inicial,  # Población inicial
        "rho": 0.6,  # Porcentaje de ocupación de corales del Reef inicial
        "Fb": 0.98,  # Proporción de Broadcast Spawning
        "Fd": 0.5,  # Proporción de Depredación
        "Pd": 0.8,  # Probabilidad de Depredación
        "k": 3,  # Número máximo de intentos para que la larva intente asentarse
        "K": 20,  # Número máximo de corales con soluciones duplicadas
        "group_subs": True,
        # Si 'True', los corales se reproducen sólo en su mismo substrato, si 'False', se reproducen con toda la población

        "stop_cond": "Neval",  # Condición de parada
        "time_limit": 4000.0,  # Tiempo límite (real, no de CPU) de ejecución
        "Ngen": 100,  # Número de generaciones
        "Neval": 15050,  # Número de evaluaciones de la función objetivo
        "fit_target": 50,  # Valor de función objetivo a alcanzar -> Ponemos 50 por poner un valor muy bajo

        "verbose": True,  # Informe periódico de cómo va el algoritmo
        "v_timer": 1,  # Tiempo entre informes generados
        "Njobs": 1,  # Número de trabajos a ejecutar en paralelo -> Como es 1, se ejecuta de forma secuencial

        "dynamic": True,
        # Determina si usar la variante dinámica del algoritmo -> Permite cambiar el tamaño de cada substrato (Mirar paper)
        "dyn_method": "success",
        # Determina la probabilidad de elegir un substrato para cada coral en la siguiente generación -> Con 'success' usa el ratio de larvas exitosas en cada generación
        "dyn_metric": "best",  # Determina cómo agregar los valores de cada substrato para obtener la medida de cada uno
        "dyn_steps": 10,  # Número de evaluaciones por cada substrato
        "prob_amp": 0.01
        # Determina cómo las diferencias entre las métricas de los substratos afectan la probabilidad de cada una -> Cuanto más pequeña, más amplifica
    }

    operators = [
        SubstrateInt("MutSample", {"method": "Gauss", "F": 1.5, "N": 5}),  # Rand Mutation -> F = Desviación Típica; N = Número de muestras a mutar
        SubstrateInt("Multipoint"),    #Multi-Point Crossover
        #SubstrateInt("BLXalpha", {"F": 0.5}),  #BLX-Alpha -> F = Alpha
        #SubstrateInt("DE/best/1", {"F": 0.7, "Cr": 0.8})   #Differential Evolution -> F = Factor de escalado de la ecuación; Cr = Prob. de Recombinación
    ]

    for v in range(numero_bases):
        if v in ind_intermediarios:
            print("Intermediario: " + str(v) + " -> Capacidad: " + str(capacidad_bases[v]))

    solution_normal = []
    if os.path.exists(Ruta_Solucion):  # Cargamos la solución
        with open(Ruta_Solucion, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                solution_normal.append(numbers)
        solution_normal = np.array(solution_normal, dtype=int)
    solution_normal = list(solution_normal)

    lista_base_indices = []
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_inter, latitudes_inter, color='green', label='Intermediarios')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    bases = puntos[:numero_bases]
    latitudes_bases, longitudes_bases = zip(*bases)
    for v in range(len(ind_intermediarios)):
        if isinstance(solution_normal[1], float):
            solution_normal[1] = list(np.zeros(numero_bases))
        base_indices = [i for i, x in enumerate(solution_normal[1]) if x == ind_intermediarios[v]]
        for j in base_indices:
            plt.plot([longitudes_bases[j], longitudes_inter[v]],[latitudes_bases[j], latitudes_inter[v]], color='yellow')
        lista_base_indices.extend(base_indices)
    for k in range(numero_supply_depots):
        SD = [i for i, v in enumerate(solution_normal[0]) if v == k]  # Sacamos bases asociadas a un SD
        if len(SD) > 0:
            for l in range(len(SD)):
                if SD[l] in ind_intermediarios:
                    plt.plot([longitudes_bases[SD[l]], longitudes_supply_depots[solution_normal[0][SD[l]]]],
                             [latitudes_bases[SD[l]], latitudes_supply_depots[solution_normal[0][SD[l]]]], color='red')
    plt.xlabel('Distancia Horizontal (px/m)')
    plt.ylabel('Distancia Vertical (px/m)')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.gca().invert_yaxis()
    plt.show()

    #Aqúi comienza el viajante para intermediarios

    Lista_Rutas_Intermediario = []
    Lista_Fitness_Intermediario = []
    Num_Inter = len(ind_intermediarios)  # Número de intermediarios
    for i in range(Num_Inter):  #Bucle para cada intermediario
        print("Intermediario: " + str(i+1))
        indices_bases_inter = [j for j, value in enumerate(solution_normal[1]) if value == ind_intermediarios[i]]   #Sacamos índices de las bases asociadas a un intermediario
        if len(indices_bases_inter) == 0:   #No hay bases asociadas a ese intermediario
            Lista_Rutas_Intermediario.append(0.0)  # Guardamos un valor nulo
            Lista_Fitness_Intermediario.append(0.0) #Guardamos un valor nulo -> Lo hacemos para no perder el orden de los intermediarios
            continue    #Siguiente iteración
        dist_bases_list_inter = []
        dist_bases_inter = [bases[v] for v in indices_bases_inter]  # Bases asociadas a un intermediario
        dist_bases_inter.append(bases[ind_intermediarios[i]])  # Añadimos también el propio intermediario
        for x in range(len(indices_bases_inter)):  # Sacamos distancias entre las bases del mismo SD
            distancia_euclidea_inter = Distancia_Base_Supply_Depot_2D(dist_bases_inter, bases[indices_bases_inter[x]])  # Obtenemos distancias de bases con otra base
            dist_bases_list_inter.append(distancia_euclidea_inter)  # Añadimos esas distancias a la lista principal -> Al final obtenemos una diagonal de 0's
        if len(indices_bases_inter) == 1: #Una base asociada a ese intermediario
            fitness_1 = 0.0
            fitness_1 += 2.0 * dist_bases_list_inter[0][len(dist_bases_list_inter)]
            Ruta_Inter = indices_bases_inter   #Sacamos la base asociada y la guardamos
            Lista_Rutas_Intermediario.append(Ruta_Inter)  # Guardamos la ruta de ese intermediario
            Lista_Fitness_Intermediario.append(fitness_1)
            continue
        flag_1 = True
        objfunc_Viajante = Fitness(len(indices_bases_inter))
        Coral_Inter = CRO_SL(objfunc_Viajante, operators, params_Inter)
        solution_Inter, Costes_Inter = Coral_Inter.optimize()
        flag_1 = False  #Usamos flags para saber cuándo entra en un viajante o en otro del CRO SL
        print("Coste Solución " + str(i) + ": " + str(Costes_Inter))
        Array_Indices = np.array(indices_bases_inter)
        Ruta_Inter = Array_Indices[solution_Inter.astype(int)]
        Lista_Rutas_Intermediario.append(Ruta_Inter)    #Guardamos la ruta de ese intermediario
        Lista_Fitness_Intermediario.append(Costes_Inter)

    #Aquí comienza el viajante normal
    Lista_Sol_Final = []
    contador_aux = 0.0
    for k in range(numero_supply_depots):
        print("SD: " + str(k + 1))
        indices_bases_SD = np.array([j for j, value in enumerate(solution_normal[0]) if value == k])  # Sacamos índices de las bases asociadas a un SD
        indices_bases_inter = np.array([j for j, value in enumerate(indices_bases_SD) if solution_normal[1][value] == numero_bases])  # Sacamos bases e inters directos a un SD
        dist_bases_list_SD = []
        bases_SD = [bases[v] for v in indices_bases_SD[indices_bases_inter]]  # Bases asociadas a un SD
        for x in range(len(indices_bases_inter)):  # Sacamos distancias entre las bases del mismo SD
            distancia_euclidea_SD = Distancia_Base_Supply_Depot_2D(bases_SD, bases[indices_bases_SD[indices_bases_inter[x]]])  # Obtenemos distancias de bases con otra base
            dist_bases_list_SD.append(distancia_euclidea_SD)  # Añadimos esas distancias a la lista principal -> Al final obtenemos una diagonal de 0's
        flag_2 = True
        objfunc = Fitness(len(indices_bases_inter))  # Función objetivo para un tamaño de vector igual al número de bases e inters asociados a un SD
        Coral = CRO_SL(objfunc,operators,params)
        solution, Costes_Viajante = Coral.optimize()
        flag_2 = False
        print("Coste Solución SD " + str(k + 1) + ": " + str(Costes_Viajante))
        contador_aux += Costes_Viajante
        Lista_Sol_Final.append(indices_bases_SD[indices_bases_inter][solution.astype(int)])  # Guardamos la ruta de la solución
    print("Media Costes SD: " + str(contador_aux / numero_supply_depots))

    #Graficamos la solución
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_inter, latitudes_inter, color='green', label='Intermediarios')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    fig.show()
    #Evolución del coste
    coste = plt.figure(figsize=(10, 6))
    plt.plot(Coral.history)
    coste.show()
    #Graficar solución
    colores = ['green', 'magenta', 'red', 'orange', 'purple', 'brown', 'pink', 'yellow', 'black', 'cyan']
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_inter, latitudes_inter, color='green', label='Intermediarios')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p',s=60,label='Puntos de Suministro')
    for v in range(len(Lista_Sol_Final)-7):
        color = colores[v % len(colores)]  # Un color por cada iteración
        plt.plot([longitudes_bases[Lista_Sol_Final[v][0]], longitudes_supply_depots[v]],
                 [latitudes_bases[Lista_Sol_Final[v][0]], latitudes_supply_depots[v]], color=color)
        for k in range(0, len(Lista_Sol_Final[v]) - 1):  # Bucle que recorre los valores
            if Lista_Sol_Final[v][k] in ind_intermediarios and Lista_Sol_Final[v][k + 1] in ind_intermediarios:  # Si los siguientes son intermediarios
                indice_lista_rutas_1, indice_lista_rutas_2 = np.where(ind_intermediarios == Lista_Sol_Final[v][k])[0],np.where(ind_intermediarios == Lista_Sol_Final[v][k + 1])[0]  # Buscamos el índice del intermediario
                plt.plot([longitudes_inter[indice_lista_rutas_1[0]], longitudes_inter[indice_lista_rutas_2[0]]],
                         [latitudes_inter[indice_lista_rutas_1[0]], latitudes_inter[indice_lista_rutas_2[0]]],color=color)
                if not isinstance(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]], float):  # Tiene bases asociadas
                    if len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) == 1:  # Sólo una base asociada
                        plt.plot([longitudes_inter[indice_lista_rutas_1[0]],
                                  longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]],
                                 [latitudes_inter[indice_lista_rutas_1[0]],
                                  latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]], color=color)
                    else:  # Varias bases asociadas
                        plt.plot([longitudes_inter[indice_lista_rutas_1[0]],
                                  longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]],
                                 [latitudes_inter[indice_lista_rutas_1[0]],
                                  latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]], color=color)
                        for l in range(len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) - 1):
                            plt.plot([longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l]],
                                      longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l + 1]]],
                                     [latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l]],
                                      latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l + 1]]],
                                     color=color)
                        plt.plot([longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][
                            len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) - 1]],
                                  longitudes_bases[Lista_Sol_Final[v][k]]],
                                 [latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][
                                     len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) - 1]],
                                  latitudes_bases[Lista_Sol_Final[v][k]]], color=color)
            elif Lista_Sol_Final[v][k] in ind_intermediarios:  # Si el primero es intermediario y el siguiente no lo es
                indice_lista_rutas_1 = np.where(ind_intermediarios == Lista_Sol_Final[v][k])[0]  # Buscamos el índice del intermediario
                if not isinstance(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]], float):  # Tiene bases asociadas
                    if len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) == 1:  # Sólo una base asociada
                        plt.plot([longitudes_inter[indice_lista_rutas_1[0]],
                                  longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]],
                                 [latitudes_inter[indice_lista_rutas_1[0]],
                                  latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]], color=color)
                    else:  # Varias bases asociadas
                        plt.plot([longitudes_inter[indice_lista_rutas_1[0]],
                                  longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]],
                                 [latitudes_inter[indice_lista_rutas_1[0]],
                                  latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]], color=color)
                        for l in range(len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) - 1):
                            plt.plot([longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l]],
                                      longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l + 1]]],
                                     [latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l]],
                                      latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l + 1]]],
                                     color=color)
                        plt.plot([longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][
                            len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) - 1]],
                                  longitudes_inter[indice_lista_rutas_1[0]]],
                                 [latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][
                                     len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) - 1]],
                                  latitudes_inter[indice_lista_rutas_1[0]]], color=color)
                plt.plot([longitudes_inter[indice_lista_rutas_1[0]], longitudes_bases[Lista_Sol_Final[v][k + 1]]],
                         [latitudes_inter[indice_lista_rutas_1[0]], latitudes_bases[Lista_Sol_Final[v][k + 1]]],
                         color=color)
            else:  # Si no es intermediario
                if Lista_Sol_Final[v][k + 1] in ind_intermediarios:  # Si el siguiente es intermediario
                    indice_lista_rutas_2 = np.where(ind_intermediarios == Lista_Sol_Final[v][k + 1])[
                        0]  # Buscamos el índice del intermediario
                    plt.plot([longitudes_bases[Lista_Sol_Final[v][k]], longitudes_inter[indice_lista_rutas_2[0]]],
                             [latitudes_bases[Lista_Sol_Final[v][k]], latitudes_inter[indice_lista_rutas_2[0]]],
                             color=color)
                elif Lista_Sol_Final[v][k + 1] not in ind_intermediarios:  # Si el siguiente es base
                    plt.plot([longitudes_bases[Lista_Sol_Final[v][k]], longitudes_bases[Lista_Sol_Final[v][k + 1]]],
                             [latitudes_bases[Lista_Sol_Final[v][k]], latitudes_bases[Lista_Sol_Final[v][k + 1]]],
                             color=color)
    plt.xlabel('Distancia Horizontal (px/m)')
    plt.ylabel('Distancia Vertical (px/m)')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.gca().invert_yaxis()
    plt.show()