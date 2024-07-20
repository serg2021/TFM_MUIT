import numpy as np
from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateInt import *
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import csv


class Fitness(AbsObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size    #Número de bases
        super().__init__(self.size, opt)

    def objective(self, solution):  #Función objetivo -> El algoritmo se encarga de hacerlo, no es como el evolutivo
        return Funcion_Fitness(distancias_euclideas, solution)

    def random_solution(self):  #Generamos una población inicial -> Solo indicamos cómo serán las soluciones y las reparamos una vez se generen, el resto lo hace el algoritmo
        Pob_Ini = list(np.random.randint(0, numero_supply_depots, size=self.size))  # Solución tipo
        Asig_Clases_SD =np.full((numero_bases, numero_clases),numero_supply_depots)  # Para cambiar la asignación de la capacidad de un recurso de una base a distintos SD
        Asig_Clases_SD = Asig_Clases_SD.tolist()
        for k in range(self.size):
            ind_clases = [i for i, v in enumerate(lista_clases_base[k]) if isinstance(v, str)]
            for l in ind_clases:
                Asig_Clases_SD[k][l] = Pob_Ini[k]
        Pob_Ini = [Pob_Ini, Asig_Clases_SD]
        if(Comprobacion_Individuo(Pob_Ini, capacidad_bases)):
            Pob_Ini = Reparacion_Mayor_Menor(Pob_Ini, capacidad_bases)
        return Pob_Ini

    def repair_solution(self, solution):    #Reparación de individuos
        for i in range(numero_bases):
            if isinstance(solution[0][i], list):
                solution[0][i] = list(set(solution[0][i]))
                if len(solution[0][i]) == 1:
                    solution[0][i] = solution[0][i][0]
                    continue
                for j in range(len(solution[0][i])):
                    if solution[0][i][j] > numero_supply_depots-1 or solution[0][i][j] < 0:
                        ind_asig = [l for l, v in enumerate(solution[1][i]) if v == solution[0][i][j]]
                        solution[0][i][j] = np.random.randint(0, numero_supply_depots)
                        for k in ind_asig:
                            solution[1][i][k] = solution[0][i][j]  # Actualizamos la asignación de clases de la base mutada
            elif solution[0][i] > numero_supply_depots-1 or solution[0][i] < 0:
                ind_asig = [j for j, v in enumerate(solution[1][i]) if v == solution[0][i]]
                solution[0][i] = np.random.randint(0, numero_supply_depots)
                for k in ind_asig:
                    solution[1][i][k] = solution[0][i]    #Actualizamos la asignación de clases de la base mutada
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
    for j in range(len(individuo[0])):
        SD = individuo[0][j]    #Saco el SD asociado a una base de la población
        if isinstance(SD, list):  # Si hay muchos SD asociados a una base
            if len(SD) == 0:
                break
            dist_SD = []
            for k in SD:  # Obtenemos todas las distancias de las bases a los SD y obtenemos la mínima
                if (k > numero_supply_depots-1 or k < 0 or isinstance(k,float)):  # Está mutando y nos da valores de SD que no pueden ser -> SOLUCIÓN:
                    k = np.random.randint(0, numero_supply_depots)  # Se genera el número a modificar
                dist_SD.append(distancias[k][j])
            min_dist = min(dist_SD)
            fitness += min_dist
        else:
            if(SD > numero_supply_depots-1 or SD < 0 or isinstance(SD, float)):   #Está mutando y nos da valores de SD que no pueden ser -> SOLUCIÓN:
                SD = np.random.randint(0,numero_supply_depots)                      # Se genera el número a modificar:
            fitness += distancias[SD][j]  # Calculo fitness buscando en la matriz de distancias la distancia asociada
    fitness = fitness/numero_bases
    return fitness

def Comprobacion_Individuo (individuo, capacidades):
    suma_comprobar = [[0 for _ in range(numero_clases)] for _ in range(numero_supply_depots)]
    for i in range(numero_supply_depots):
        indices_bases = []
        for j, value in enumerate(individuo[0]):
            if isinstance(value, list):
                # Si el elemento es una sublista, verificamos si tiene el SD
                if len(value) != len(set(value)):
                    return True
                if i in value:
                    indices_bases.append(j)
            else:
                # Si el elemento no es una sublista, verificamos si tiene el SD
                if value == i:
                    indices_bases.append(j)
        for j in range(len(indices_bases)):
            ind_clases = [k for k, v in enumerate(lista_clases_base[indices_bases[j]]) if isinstance(v, str)]  # Sacamos qué clases tiene esa base
            for k in ind_clases:  # Bucle para cada clase de esa base
                if individuo[1][indices_bases[j]][k] == i:  # Comprobamos que para esa clase esté asignado el SD correspondiente
                    comprobar_capacidades = capacidades[indices_bases[j]][k]
                    suma_comprobar[i][k] += comprobar_capacidades
        Caps_Clase_Comprobar = [t for t, value in enumerate(suma_comprobar[i]) if value > 200 / numero_clases]
        if len(Caps_Clase_Comprobar) > 0:  # Si para un SD, se supera el umbral de al menos una clase... -> Reparación
            return True
    Caps_Comprobar = [ind_cap for ind_cap, j in enumerate(suma_comprobar) if sum(j) > 200]
    if len(Caps_Comprobar) > 0:
        return True

def Reparacion_Mayor_Menor (individuo, capacidades): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
    capacidades_sd = [[0 for _ in range(numero_clases)] for _ in range(numero_supply_depots)]
    suma_capacidades = [[0 for _ in range(numero_clases)] for _ in range(numero_supply_depots)]
    SD_ind = 0
    while True:
        indices_bases_reparar = []
        for j, value in enumerate(individuo[0]):  # Sacamos bases asociadas a un SD
            if isinstance(value, list):
                # Si el elemento es una sublista, verificamos si tiene el SD
                if SD_ind in value:
                    indices_bases_reparar.append(j)
            else:
                # Si el elemento no es una sublista, verificamos si tiene el SD
                if value == SD_ind:
                    indices_bases_reparar.append(j)
        for s in range(numero_clases):  # Inicializamos a 0 todas las sumas de capacidades para la comprobación
            suma_capacidades[SD_ind][s] = 0
        for j in range(len(indices_bases_reparar)):
            ind_clases = [k for k, v in enumerate(lista_clases_base[indices_bases_reparar[j]]) if isinstance(v, str)]  # Sacamos qué clases tiene esa base
            for k in ind_clases:  # Bucle para cada clase de esa base
                if individuo[1][indices_bases_reparar[j]][k] == SD_ind:  # Comprobamos que para esa clase esté asignado el SD correspondiente
                    capacidades_sd_i = capacidades[indices_bases_reparar[j]][k]
                    capacidades_sd[SD_ind][k] = capacidades_sd_i
                    suma_capacidades[SD_ind][k] += capacidades_sd[SD_ind][k]  # Almacenamos todas las sumas de las capacidades en un array
        Caps_Clase_Comprobar = [t for t, value in enumerate(suma_capacidades[SD_ind]) if value > 200 / numero_clases]
        if len(Caps_Clase_Comprobar) > 0:
            while True:
                k_2 = np.argsort(suma_capacidades[SD_ind])[::-1]
                k_1 = k_2[0]  # Solucionamos aquella capacidad que sea mas grande
                salir = False
                while True:
                    while True:
                        SDs = list(np.arange(numero_supply_depots))  # Sacamos lista de SDs
                        while True:
                            k_3 = random.choice(SDs)  # Elegimos uno al azar
                            if k_3 == SD_ind:
                                continue
                            else:
                                break
                        indices_bases_SD = []  # Bases asociadas al SD de la Cap más grande
                        indices_resto_bases = []  # Bases asociadas al otro SD
                        for j, value in enumerate(individuo[0]):  # Sacamos bases asociadas a un SD
                            if isinstance(value, list):
                                # Si el elemento es una sublista, verificamos si tiene el SD
                                if SD_ind in value:  # Tiene el SD asociado en el tipo de recurso que está dando problemas
                                    if individuo[1][j][k_1] == SD_ind:
                                        indices_bases_SD.append(j)
                                if k_3 in value:
                                    if individuo[1][j][k_1] == k_3:
                                        indices_resto_bases.append(j)
                            else:
                                # Si el elemento no es una sublista, verificamos si tiene el SD
                                if value == SD_ind:
                                    if individuo[1][j][k_1] == SD_ind:
                                        indices_bases_SD.append(j)
                                if value == k_3:
                                    if individuo[1][j][k_1] == k_3:
                                        indices_resto_bases.append(j)
                        capacidades_bases_SD_ordenados = list(np.argsort([capacidades[i][k_1] for i in indices_bases_SD])[::-1])
                        # Ordenamos índices de capacidades de esa clase de mayor a menor
                        indices_bases_SD_ordenados = [indices_bases_SD[i] for i in capacidades_bases_SD_ordenados]
                        # Ordenamos los índices de las bases al SD de la Cap más grande de mayor a menor
                        indice_base_1 = indices_bases_SD_ordenados[0]  # Cogemos la base con mayor cap de esa clase

                        lista_filtrada = [v for v in indices_resto_bases if capacidades[v][k_1] < capacidades[indice_base_1][k_1] and capacidades[v][k_1] != 0]
                        if lista_filtrada:
                            indice_base_aleatoria_2 = random.choice(lista_filtrada)
                        else:
                            SD_mismos_recursos = [v for i, v in enumerate(SDs) if v != SD_ind and v != k_3]
                            lista_ind = []
                            while True: #Cotejamos que haya bases de la misma clase en otros SD
                                while True:
                                    k_4 = random.choice(SD_mismos_recursos)  # Jugamos con uno de los SD con mismos recursos que las bases
                                    if k_4 == SD_ind:
                                        continue
                                    else:
                                        break
                                indices_resto_bases = []  # Bases asociadas al otro SD
                                for j, value in enumerate(individuo[0]):  # Sacamos bases asociadas a un SD
                                    if isinstance(value, list):
                                        # Si el elemento es una sublista, verificamos si tiene el SD
                                        if k_4 in value and k_4 not in lista_ind:
                                            if individuo[1][j][k_1] == k_4:
                                                indices_resto_bases.append(j)
                                    else:
                                        # Si el elemento no es una sublista, verificamos si tiene el SD
                                        if value == k_4 and value not in lista_ind:
                                            if individuo[1][j][k_1] == k_4:
                                                indices_resto_bases.append(j)
                                lista_filtrada = [v for v in indices_resto_bases if capacidades[v][k_1] < capacidades[indice_base_1][k_1] and capacidades[v][k_1] != 0]
                                if lista_filtrada:
                                    k_3 = k_4  # Actualizamos el SD a reemplazar con otro aleatorio si el primero (el k_3) no funciona
                                    break
                                else:  # Si no hay, la añadimos a una lista
                                    lista_ind.append(k_4)
                                    if len(lista_ind) == len(SD_mismos_recursos):  # Cuando el tamaño de lista sea igual que SD_mismos_recursos...
                                        e = random.randint(0, 5)
                                        f = indices_bases_SD_ordenados[0:e]
                                        for i in f:
                                            if isinstance(individuo[0][i],list):  # Si es una sublista, buscamos índice donde esté el valor de ese SD
                                                indice = [j for j, v in enumerate(individuo[0][i]) if v == SD_ind]
                                                indice_2 = [j for j, v in enumerate(individuo[0][i]) if v == k_3]
                                                if len(indice_2) > 0:
                                                    break
                                                for l in indice:
                                                    individuo[0][i][l] = k_3  # Lo cambiamos por el otro SD
                                                individuo[0][i] = list(set(individuo[0][i]))
                                                if len(individuo[0][i]) == 1:
                                                    individuo[0][i] = individuo[0][i][0]
                                                ind_aux_2 = individuo[1][i].index(SD_ind)
                                                individuo[1][i][ind_aux_2] = k_3
                                            else:
                                                indice_asig = [j for j, v in enumerate(individuo[1][i]) if v != numero_supply_depots]
                                                if len(indice_asig) == 1:
                                                    individuo[1][i][indice_asig[0]] = k_3
                                                    individuo[0][i] = k_3  # ... Descargamos algunas bases del SD que nos da problemas sobre el otro (k_3)
                                                elif len(indice_asig) > 1:
                                                    g = random.choice(indice_asig)
                                                    individuo[1][i][g] = k_3
                                                    individuo[0][i] = [individuo[0][i]]
                                                    individuo[0][i].append(k_3)
                                        break
                                    else:
                                        continue
                            if len(lista_ind) == len(SD_mismos_recursos):
                                indices_bases_SD = []  # Bases asociadas al SD de la Cap más grande
                                for j, value in enumerate(individuo[0]):  # Sacamos bases asociadas a un SD
                                    if isinstance(value, list):
                                        # Si el elemento es una sublista, verificamos si tiene el SD
                                        if SD_ind in value:
                                            indices_bases_SD.append(j)
                                    else:
                                        # Si el elemento no es una sublista, verificamos si tiene el SD
                                        if value == SD_ind:
                                            indices_bases_SD.append(j)
                                for s in range(numero_clases):  # Inicializamos a 0 todas las sumas de capacidades para la comprobación
                                    suma_capacidades[SD_ind][s] = 0
                                for j in range(len(indices_bases_SD)):
                                    ind_clases = [k for k, v in enumerate(lista_clases_base[indices_bases_SD[j]]) if isinstance(v, str)]  # Sacamos qué clases tiene esa base
                                    for t in ind_clases:  # Bucle para cada clase de esa base
                                        if individuo[1][indices_bases_SD[j]][t] == SD_ind:  # Comprobamos que para esa clase esté asignado el SD correspondiente
                                            capacidades_sd_i = capacidades[indices_bases_SD[j]][t]
                                            capacidades_sd[SD_ind][t] = capacidades_sd_i
                                            suma_capacidades[SD_ind][t] += capacidades_sd[SD_ind][t]  # Almacenamos todas las sumas de las capacidades en un array
                                if suma_capacidades[SD_ind][k_1] > 200 / numero_clases:
                                    continue
                                else:
                                    salir = True
                                    break
                            indice_base_aleatoria_2 = random.choice(indices_resto_bases)    #Elección aleatoria de la base del resto de bases


                        # Comprobamos si son listas de SD los índices cogidos
                        # Si lo son alguno (o los dos) -> Comprobamos que no tengan el SD que vaya a ponerle el otro
                        if isinstance(individuo[0][indice_base_1], list) or isinstance(individuo[0][indice_base_aleatoria_2], list):
                            if isinstance(individuo[0][indice_base_1], list) and isinstance(individuo[0][indice_base_aleatoria_2], list):
                                p = [x for x, v in enumerate(individuo[0][indice_base_1]) if v == k_3]
                                q = [x for x, v in enumerate(individuo[0][indice_base_aleatoria_2]) if v == SD_ind]
                                if len(p) == 0 and len(q) == 0:
                                    break
                                elif len(p) == 1:
                                    ind_aux = individuo[0][indice_base_1].index(k_3)
                                    individuo[0][indice_base_1].pop(ind_aux)
                                    if len(individuo[0][indice_base_1]) == 1:
                                        individuo[0][indice_base_1] = individuo[0][indice_base_1][0]
                                    break
                                elif len(q) == 1:
                                    ind_aux = individuo[0][indice_base_aleatoria_2].index(SD_ind)
                                    individuo[0][indice_base_aleatoria_2].pop(ind_aux)
                                    if len(individuo[0][indice_base_aleatoria_2]) == 1:
                                        individuo[0][indice_base_aleatoria_2] = individuo[0][indice_base_aleatoria_2][0]
                                    break

                            elif isinstance(individuo[0][indice_base_1], list):
                                p = [x for x, v in enumerate(individuo[0][indice_base_1]) if v == k_3]
                                if len(p) == 0:
                                    break
                                elif len(p) == 1:
                                    ind_aux = individuo[0][indice_base_1].index(k_3)
                                    individuo[0][indice_base_1].pop(ind_aux)
                                    if len(individuo[0][indice_base_1]) == 1:
                                        individuo[0][indice_base_1] = individuo[0][indice_base_1][0]
                                    break
                            elif isinstance(individuo[0][indice_base_aleatoria_2], list):
                                q = [x for x, v in enumerate(individuo[0][indice_base_aleatoria_2]) if v == SD_ind]
                                if len(q) == 0:
                                    break
                                elif len(q) == 1:
                                    ind_aux = individuo[0][indice_base_aleatoria_2].index(SD_ind)
                                    individuo[0][indice_base_aleatoria_2].pop(ind_aux)
                                    if len(individuo[0][indice_base_aleatoria_2]) == 1:
                                        individuo[0][indice_base_aleatoria_2] = individuo[0][indice_base_aleatoria_2][0]
                                    break
                        else:
                            break

                    if salir:
                        break
                    # Hacemos el intercambio de SD según el tipo de dato que tenga la base: int o list

                    if isinstance(individuo[0][indice_base_1], (int, np.integer)) and isinstance(individuo[0][indice_base_aleatoria_2], (int, np.integer)):
                        comprob_asig_1 = [i for i, v in enumerate(individuo[1][indice_base_1]) if v != numero_supply_depots]
                        comprob_asig_2 = [i for i, v in enumerate(individuo[1][indice_base_aleatoria_2]) if v != numero_supply_depots]
                        if len(comprob_asig_1) == 1 and len(comprob_asig_2) == 1:  # Si sólo tienen 1 clase las 2 bases... Las podemos intercambiar
                            if comprob_asig_1[0] == k_1 and comprob_asig_2[0] == k_1:
                                individuo[1][indice_base_1][comprob_asig_1[0]], individuo[1][indice_base_aleatoria_2][comprob_asig_2[0]] = individuo[1][indice_base_aleatoria_2][comprob_asig_2[0]], individuo[1][indice_base_1][comprob_asig_1[0]]
                                individuo[0][indice_base_1], individuo[0][indice_base_aleatoria_2] = individuo[0][indice_base_aleatoria_2],individuo[0][indice_base_1]
                        else:  # Si no, añadimos el SD a la lista de SDs de esa base
                            if len(comprob_asig_1) > 1:
                                if k_1 in comprob_asig_1:
                                    individuo[1][indice_base_1][k_1] = individuo[0][indice_base_aleatoria_2]  # Actualizamos la asignación de esa clase al nuevo SD
                                else:
                                    individuo[1][indice_base_1][random.choice(comprob_asig_1)] = individuo[0][indice_base_aleatoria_2]
                                individuo[0][indice_base_1] = [individuo[0][indice_base_1]]  # Lo pasamos a lista
                                individuo[0][indice_base_1].append(individuo[0][indice_base_aleatoria_2])  # Le añadimos el nuevo SD
                                if k_1 in comprob_asig_2:
                                    if len(comprob_asig_2) > 1:
                                        individuo[1][indice_base_aleatoria_2][k_1] = SD_ind  # Actualizamos la asignación de esa clase al viejo SD
                                        individuo[0][indice_base_aleatoria_2] = [individuo[0][indice_base_aleatoria_2]]  # Lo pasamos a lista
                                        individuo[0][indice_base_aleatoria_2].append(SD_ind)  # Le añadimos el viejo SD
                                    elif len(comprob_asig_2) == 1:
                                        if comprob_asig_2[0] == k_1:
                                            individuo[1][indice_base_aleatoria_2][k_1] = SD_ind  # Actualizamos la asignación de esa clase al nuevo SD
                                            individuo[0][indice_base_aleatoria_2] = SD_ind

                            elif len(comprob_asig_1) == 1:
                                if individuo[1][indice_base_1][k_1] == individuo[1][indice_base_1][comprob_asig_1[0]] and individuo[1][indice_base_1][k_1] != numero_supply_depots:
                                    individuo[1][indice_base_1][k_1] = individuo[0][indice_base_aleatoria_2]  # Actualizamos la asignación de esa clase al nuevo SD
                                else:
                                    individuo[1][indice_base_1][comprob_asig_1[0]] = individuo[0][indice_base_aleatoria_2]
                                SD_aux = individuo[0][indice_base_1]
                                individuo[0][indice_base_1] = individuo[0][indice_base_aleatoria_2]
                                if k_1 in comprob_asig_2:
                                    if len(comprob_asig_2) > 1:
                                        individuo[1][indice_base_aleatoria_2][k_1] = SD_aux  # Actualizamos la asignación de esa clase al viejo SD
                                        individuo[0][indice_base_aleatoria_2] = [individuo[0][indice_base_aleatoria_2]]  # Lo pasamos a lista
                                        individuo[0][indice_base_aleatoria_2].append(SD_aux)  # Le añadimos el viejo SD
                                    elif len(comprob_asig_2) == 1:
                                        if comprob_asig_2[0] == k_1:
                                            individuo[1][indice_base_aleatoria_2][k_1] = SD_aux  # Actualizamos la asignación de esa clase al nuevo SD
                                            individuo[0][indice_base_aleatoria_2] = SD_aux

                    elif isinstance(individuo[0][indice_base_1], list) and isinstance(individuo[0][indice_base_aleatoria_2], list):
                        if individuo[1][indice_base_1][k_1] != numero_supply_depots:
                            individuo[1][indice_base_1][k_1] = k_3  # Actualizamos la asignación de esa clase al nuevo SD
                            comprob_asig_1 = [i for i, v in enumerate(individuo[1][indice_base_1]) if v == SD_ind]
                            # Si al actualizar la asignación sigue estando el SD que estamos analizando -> No hacemos nada
                            # Sin embargo, si ya no aparece, tenemos que quitarlo de la lista
                            if len(comprob_asig_1) == 0:
                                ind_cambio = individuo[0][indice_base_1].index(SD_ind)
                                individuo[0][indice_base_1].pop(ind_cambio)  # Le quitamos ese SD
                            individuo[0][indice_base_1].append(k_3)  # Le añadimos el nuevo SD SIEMPRE

                        if individuo[1][indice_base_aleatoria_2][k_1] != numero_supply_depots:
                            individuo[1][indice_base_aleatoria_2][k_1] = SD_ind  # Actualizamos la asignación de esa clase al nuevo SD
                            comprob_asig_2 = [i for i, v in enumerate(individuo[1][indice_base_aleatoria_2]) if v == k_3]
                            # Si al actualizar la asignación sigue estando el SD que estamos analizando -> No hacemos nada
                            # Sin embargo, si ya no aparece, tenemos que quitarlo de la lista
                            if len(comprob_asig_2) == 0:
                                ind_cambio = individuo[0][indice_base_aleatoria_2].index(k_3)
                                individuo[0][indice_base_aleatoria_2].pop(ind_cambio)  # Le quitamos ese SD
                            individuo[0][indice_base_aleatoria_2].append(SD_ind)  # Le añadimos el nuevo SD SIEMPRE

                    elif isinstance(individuo[0][indice_base_1], list) or isinstance(individuo[0][indice_base_aleatoria_2], list):
                        if isinstance(individuo[0][indice_base_1], list):
                            if individuo[1][indice_base_1][k_1] != numero_supply_depots:
                                individuo[1][indice_base_1][k_1] = individuo[0][indice_base_aleatoria_2]
                                comprob_asig_1 = [i for i, v in enumerate(individuo[1][indice_base_1]) if v == SD_ind]
                                if len(comprob_asig_1) == 0:
                                    ind_cambio = individuo[0][indice_base_1].index(SD_ind)
                                    individuo[0][indice_base_1].pop(ind_cambio)  # Le quitamos ese SD
                                individuo[0][indice_base_1].append(individuo[0][indice_base_aleatoria_2])

                            if individuo[1][indice_base_aleatoria_2][k_1] != numero_supply_depots:
                                individuo[1][indice_base_aleatoria_2][k_1] = SD_ind
                                comprob_asig_1 = [i for i, v in enumerate(individuo[1][indice_base_aleatoria_2]) if v != numero_supply_depots]
                                if len(comprob_asig_1) == 1:
                                    individuo[0][indice_base_aleatoria_2] = SD_ind
                                elif len(comprob_asig_1) > 1:
                                    individuo[0][indice_base_aleatoria_2] = [individuo[0][indice_base_aleatoria_2]]
                                    individuo[0][indice_base_aleatoria_2].append(SD_ind)

                        elif isinstance(individuo[0][indice_base_aleatoria_2], list):
                            if individuo[1][indice_base_aleatoria_2][k_1] != numero_supply_depots:
                                individuo[1][indice_base_aleatoria_2][k_1] = individuo[0][indice_base_1]
                                comprob_asig_2 = [i for i, v in enumerate(individuo[1][indice_base_aleatoria_2]) if v == k_3]
                                if len(comprob_asig_2) == 0:
                                    ind_cambio = individuo[0][indice_base_aleatoria_2].index(k_3)
                                    individuo[0][indice_base_aleatoria_2].pop(ind_cambio)  # Le quitamos ese SD
                                individuo[0][indice_base_aleatoria_2].append(individuo[0][indice_base_1])

                            if individuo[1][indice_base_1][k_1] != numero_supply_depots:
                                individuo[1][indice_base_1][k_1] = k_3
                                comprob_asig_2 = [i for i, v in enumerate(individuo[1][indice_base_1]) if v != numero_supply_depots]
                                if len(comprob_asig_2) == 1:
                                    individuo[0][indice_base_1] = k_3
                                elif len(comprob_asig_2) > 1:
                                    individuo[0][indice_base_1] = [individuo[0][indice_base_1]]
                                    individuo[0][indice_base_1].append(k_3)

                    for j in range(len(individuo[0])):  #Aseguramos que no se repitan SD por base
                        if isinstance(individuo[0][j], list):
                            individuo[0][j] = list(set(individuo[0][j]))
                    indices_bases_SD = []  # Bases asociadas al SD de la Cap más grande
                    for j, value in enumerate(individuo[0]):  # Sacamos bases asociadas a un SD
                        if isinstance(value, list):
                            # Si el elemento es una sublista, verificamos si tiene el SD
                            if SD_ind in value:
                                indices_bases_SD.append(j)
                        else:
                            # Si el elemento no es una sublista, verificamos si tiene el SD
                            if value == SD_ind:
                                indices_bases_SD.append(j)
                    for s in range(numero_clases):  # Inicializamos a 0 todas las sumas de capacidades para la comprobación
                        suma_capacidades[SD_ind][s] = 0
                    for j in range(len(indices_bases_SD)):
                        ind_clases = [k for k, v in enumerate(lista_clases_base[indices_bases_SD[j]]) if isinstance(v, str)]  # Sacamos qué clases tiene esa base
                        for t in ind_clases:  # Bucle para cada clase de esa base
                            if individuo[1][indices_bases_SD[j]][t] == SD_ind:  # Comprobamos que para esa clase esté asignado el SD correspondiente
                                capacidades_sd_i = capacidades[indices_bases_SD[j]][t]
                                capacidades_sd[SD_ind][t] = capacidades_sd_i
                                suma_capacidades[SD_ind][t] += capacidades_sd[SD_ind][t]  # Almacenamos todas las sumas de las capacidades en un array
                    if suma_capacidades[SD_ind][k_1] > 200 / numero_clases:
                        continue
                    else:
                        break
                Caps_Clase_Comprobar = [t for t, value in enumerate(suma_capacidades[SD_ind]) if value > 200 / numero_clases]
                if len(Caps_Clase_Comprobar) == 0:  # Comprobamos las clases para el SD -> Si no superan el umbral, seguimos
                    break
            Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if sum(j) > 200]  # Comprobamos en qué SD's se han superado la capacidad
            if len(Caps_SD_Superadas) > 0:
                SD_ind = Caps_SD_Superadas[0]  # Elegimos el primer SD que nos salga de la lista
            else:
                SD_ind = 0  # Si no hay ninguno, hacemos un barrido por todos los SD
        else:
            SD_ind += 1
            if SD_ind == numero_supply_depots:
                Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if sum(j) > 200]  # Comprobamos en qué SD's se han superado la capacidad
                if len(Caps_SD_Superadas) > 0:
                    SD_ind = 0
                else:
                    break
    for i in range(len(individuo[0])):  #La asignación de clases está bien hecha, que es lo importante. Con esto aseguramos que la solución coincida
        indice_3 = [v for j,v in enumerate(individuo[1][i]) if v != numero_supply_depots]
        indice_3 = list(set(indice_3))
        if len(indice_3) == 1:
            individuo[0][i] = indice_3[0]
        else:
            individuo[0][i] = []
            for k in indice_3:
                individuo[0][i].append(k)
    return individuo

if __name__ == "__main__":

    Pob_Actual = []
    Costes = []
    poblacion_inicial = 100
    Num_Gen = 100
    numero_bases = 200
    numero_supply_depots = 30
    capacidad_maxima = 20
    numero_clases = 5
    Ruta_Puntos = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Bases_SD.csv")
    Ruta_Capacidades = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Cap_Bases.csv")
    Ruta_Clases_Bases = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Clases_Bases.csv")
    Ruta_Caps_Clases_Bases = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Caps_Clases_Bases.csv")
    Ruta_Caps_Clases_SD = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Caps_Clases_SD.csv")
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
    supply_depots = puntos[-numero_supply_depots:]
    bases = puntos[:numero_bases]
    longitudes_bases, latitudes_bases = zip(*bases)
    if not os.path.exists(Ruta_Capacidades):
        capacidad_bases = np.random.randint(1, capacidad_maxima, size=len(bases))
        np.savetxt(Ruta_Capacidades, capacidad_bases, delimiter=',')
    else:
        capacidad_bases = []
        with open(Ruta_Capacidades, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = float(fila[0])
                capacidad_bases.append(int(numbers))
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    longitudes_supply_depots, latitudes_supply_depots = zip(*supply_depots)
    capacidad_supply_depots = list(np.full(numero_supply_depots,200))

    clases = ['A', 'B', 'C', 'D', 'E']
    if not os.path.exists(Ruta_Clases_Bases) or not os.path.exists(Ruta_Caps_Clases_Bases):
        lista_clases_base = []
        for i in range(numero_bases):
            while True:
                vector_clase = list(np.zeros(numero_clases))
                caps_bases_clases = list(np.zeros(numero_clases, dtype=int))
                cont = 0
                for j in range(len(clases)):
                    if random.choice([True, False]):  # Generamos aleatoriamente bases que pidan un conjunto de recursos distintos entre sí
                        vector_clase[j] = clases[j]
                        cont += 1
                if any(isinstance(x, str) for x in
                       vector_clase):  # Si me saca un array a 0's, le obligo a volver a hacerlo hasta que sea diferente
                    break
            if capacidad_bases[i] < cont:
                capacidad_bases[i] = cont  # Hacemos esto para que, si la capacidad de la base es tan baja que no da para todas las clases, forzamos a que tenga más
            if capacidad_bases[i] % cont == 0:  # Quiere decir que la capacidad se reparte de forma equitativa entre las clases
                for k in range(len(clases)):
                    if vector_clase[k] == 0:
                        caps_bases_clases[k] = 0
                    else:
                        caps_bases_clases[k] = int(capacidad_bases[i] / cont)
            elif capacidad_bases[i] % cont > 0:  # Quiere decir que la capacidad de una clase será 1 unidad mayor
                indices_clases = [j for j, v in enumerate(vector_clase) if isinstance(v, str)]  # Sacamos índices donde hay clases
                ind = random.choice(indices_clases)  # Elegimos uno aleatoriamente
                caps_bases_clases[ind] = int(capacidad_bases[i] / cont) + (capacidad_bases[i] % cont)  # Para esa clase habrá un poco más que para el resto
                for k in range(len(clases)):
                    if vector_clase[k] == 0:
                        caps_bases_clases[k] = 0
                    elif k != ind:
                        caps_bases_clases[k] = int(capacidad_bases[i] / cont)
            capacidad_bases[i] = tuple(caps_bases_clases)
            lista_clases_base.append(vector_clase)
        capacidad_bases = np.array(capacidad_bases)
        np.savetxt(Ruta_Caps_Clases_Bases, capacidad_bases, delimiter=',')
        with open(Ruta_Clases_Bases, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(lista_clases_base)
    else:
        lista_clases_base = []
        capacidad_bases = []
        with open(Ruta_Clases_Bases, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) if x == '0.0' else x for x in fila]
                lista_clases_base.append(numbers)
        with open(Ruta_Caps_Clases_Bases, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                capacidad_bases.append(numbers)
            capacidad_bases = np.array(capacidad_bases)

    if not os.path.exists(Ruta_Caps_Clases_SD):
        lista_clases_SD = []
        for i in range(numero_supply_depots):
            caps_SD_clases = []
            vector_clase = list(np.zeros(numero_clases, dtype=int))
            for j in range(len(clases)):  # Todos los SD podrán dar todo tipo de recursos
                vector_clase[j] = clases[j]
                caps_SD_clases.append(200 / len(clases))
            lista_clases_SD.append(vector_clase)
            capacidad_supply_depots[i] = tuple(caps_SD_clases)  # Como un SD tiene un Cap Máxima, cada clase también lo tendrá
        np.savetxt(Ruta_Caps_Clases_SD, capacidad_supply_depots, delimiter=',')
    else:
        lista_clases_SD = []
        with open(Ruta_Caps_Clases_SD, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                lista_clases_SD.append(numbers)
            lista_clases_SD = np.array(lista_clases_SD)

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
        if isinstance(solution[0][j], list):
            solution[0][j] = list(set(solution[0][j]))
        print("Base " + str(j) + "-> SD: " + str(solution[0][j]))
    print("Coste final: " + str(obj_value))
    # Graficar el mapa y los puntos
    fig_1 = plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p',label='Puntos de Suministro')
    fig_1.show()
    #Evolución del coste de una de las rutas
    coste = plt.figure(figsize=(10, 6))
    plt.plot(Coral.history)
    coste.show()
    #Graficar solución
    colores = ['green', 'blue', 'red', 'orange', 'purple']  # Lista de colores para cada tipo de recurso
    fig = plt.figure(figsize=(10, 6))
    ejes = fig.add_subplot(111)  # Creamos ejes en la figura (1 fila, 1 columna y 1 cuadrícula) -> Necesarios para dibujar los puntos multicolor
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p',label='Puntos de Suministro')
    for k in range(numero_supply_depots):
        SD = []
        for j, value in enumerate(solution[0]):  # Obtenemos lista de índices de las bases de la solución que tienen el SD asociado
            if isinstance(value, list):
                # Si el elemento es una sublista, verificamos si tiene el SD
                if k in value:
                    SD.append(j)
            else:
                # Si el elemento no es una sublista, verificamos si tiene el SD
                if value == k:
                    SD.append(j)
        for t in SD:  # Bucle para pintar puntos multicolor
            clases_recurso = [i for i, v in enumerate(lista_clases_base[t]) if isinstance(v, str)]  # Sacamos índices de la lista de clases de la base
            lista_colores = [colores[j] for j, x in enumerate(clases_recurso)]  # Obtenemos los colores para cada clase dentro del punto que dibujaremos
            angulo_color = 360 / len(lista_colores)
            for s, color in enumerate(lista_colores):
                angulo_inicio = s * angulo_color
                seccion_circulo = patches.Wedge((longitudes_bases[t], latitudes_bases[t]), 2, angulo_inicio, angulo_inicio + angulo_color, color=color)
                ejes.add_patch(seccion_circulo)
        ejes.set_xlim(-5, 185)  # Limitar el eje x de 0 a 180
        ejes.set_ylim(-5, 185)  # Limitar el eje y de 0 a 180
        ejes.set_aspect('equal')  # Para que los puntos multicolor no queden ovalados, sino circulares
        if len(SD) > 0:  # Porque puede haber bases que no tengan asociado el SD de la iteración que toca
            aux = random.choice(SD)  # Punto del que saldrán las líneas a los SD
            if isinstance(solution[0][aux], list):  # Si un punto cualquiera está asociado a más de un SD
                for i in range(len(solution[0][aux])):
                    plt.plot([longitudes_bases[aux], longitudes_supply_depots[solution[0][aux][i]]],[latitudes_bases[aux], latitudes_supply_depots[solution[0][aux][i]]], color='red')
            else:
                plt.plot([longitudes_bases[aux], longitudes_supply_depots[solution[0][aux]]],[latitudes_bases[aux], latitudes_supply_depots[solution[0][aux]]], color='red')
        else:
            continue
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()