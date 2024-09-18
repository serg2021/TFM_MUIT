# -*- coding: utf-8 -*-
"""
@author: Sergio

1º -> Activamos el entorno de conda -> conda activate TFM_MUIT
2º -> Podemos ejecutar el código sin problemas
"""
import numpy as np
from PyCROSL.CRO_SL import CRO_SL
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio
import copy
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
    
    def PoblacionInicial(self, Capacidades, Lista, Fil=None, Col=None, Num_Max=None):
        if Fil == None:
            Fil = self.Num_Individuos
        if Col == None:
            Col = self.Tam_Individuos
        if Num_Max == None:
            Num_Max = self.Num_Max
        Pob_Ini = []
        for j in range(Fil):    #Comprobamos todos los individuos y los reparamos si estuvieran mal
            Asig_Clases_SD = np.full((numero_bases, numero_clases),numero_supply_depots)  # Para cambiar la asignación de la capacidad de un recurso de una base a distintos SD
            Ind_Aux = list(np.random.randint(0, Num_Max, size=Col))  # Son los índices de los SD asignados a cada base
            for k in range(Col):
                ind_clases = [i for i, v in enumerate(lista_clases_base[k]) if isinstance(v, str)]
                for l in ind_clases:
                    Asig_Clases_SD[k][l] = Ind_Aux[k]
            if(self.Comprobacion_Individuo(Ind_Aux, Asig_Clases_SD, Capacidades)):
                Ind_Aux, Asig_Clases_SD = self.Reparacion_Mayor_Menor(Ind_Aux, Asig_Clases_SD, Capacidades)
                if (self.Comprobacion_Individuo(Ind_Aux, Asig_Clases_SD, Capacidades)):
                    Ind_Aux, Asig_Clases_SD = self.Reparacion_Mayor_Menor(Ind_Aux, Asig_Clases_SD, Capacidades)
            Pob_Ini.append(Ind_Aux)
            Lista.append(Asig_Clases_SD)
        return Pob_Ini, Lista

    def Seleccion(self, poblacion_inicial,lista_asig, coste):
        index = list(np.argsort(coste))
        coste_ordenado = np.sort(coste)
        poblacion_actual = [poblacion_inicial[v] for i,v in enumerate(index)]   #La población tendrá más soluciones que la inicial debido al cruce
        poblacion_actual = poblacion_actual[0:self.Num_Padres]    #Nos quedamos con los mejores individuos
        lista_asig_fin = [lista_asig[v] for i, v in enumerate(index)]  # La población tendrá más soluciones que la inicial debido al cruce
        lista_asig_fin = lista_asig_fin[0:self.Num_Padres]  # Nos quedamos con los mejores individuos
        return poblacion_actual, lista_asig_fin, coste_ordenado

    def Cruce (self, poblacion, lista, capacidades, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        #Indice_Seleccionado = []
        Indices_Validos = list(np.arange(self.Num_Padres))

        for i in range(self.Num_Individuos - self.Num_Padres): #Bucle para generar HIJOS
            Indice_Padres = random.sample(Indices_Validos, 2)
            #Indice_Padres = random.sample([j for j in Indices_Validos if j not in Indice_Seleccionado], 2)            # Se elige aleatoriamente el indice de los padres
            #Indice_Seleccionado.extend(Indice_Padres)   #Guardamos los índices elegidos para que no los vuelva a repetir en la siguiente iteración
            Padre1 = poblacion[Indice_Padres[0]]                              # Se coge el padre 1
            Padre2 = poblacion[Indice_Padres[1]]                              # Se coge el padre 2
            Hijo = copy.deepcopy(Padre1)                                            # El hijo va a ser una copia del padre 1
            Asig_Hijo = copy.deepcopy(lista[Indice_Padres[0]])   #Copiamos a su vez las asignaciones de clases a SDs del padre 1
            vector = 1*(np.random.rand(self.Tam_Individuos) > self.Prob_Cruce)  # Se genera un vector para seleccionar los genes del padre 2
            if any(vector):
                for j in np.where(vector==1)[0]:
                    Hijo[j] = copy.deepcopy(Padre2[j])       # Los genes seleccionados del padre 2 pasan al hijo -> Hacemos copia profunda para no variar Padre2
                    Asig_Hijo[j] = copy.deepcopy(lista[Indice_Padres[1]][j])  #Vamos actualizando la lista de asignaciones para que sea la misma
            if np.random.rand() < self.Prob_Mutacion:                           # Se comprueba si el hijo va a mutar
                Hijo, Asig_Hijo = self.Mutacion(Hijo,Asig_Hijo, Num_Max)
            if(self.Comprobacion_Individuo(Hijo,Asig_Hijo, capacidades)):                    # Se comprueba si hay que reparar el hijo
                 Hijo, Asig_Hijo = self.Reparacion_Mayor_Menor(Hijo,Asig_Hijo, capacidades)
            #poblacion = np.insert(poblacion,self.Num_Padres+i,Hijo, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado
            poblacion.append(Hijo)  #Añadimos otro elemento a la población
            lista.append(Asig_Hijo)
        return poblacion, lista

    def Mutacion (self, individuo,asig, Num_Max=None):
        aux1 = np.random.randint(0, len(individuo))                         # Se genera número aleatorio para ver la posición que muta
        if isinstance(individuo[aux1], list):  #Si la base a mutar está asociada a varios SD, cambiamos uno de esos SD
            while True:
                aux2 = np.random.randint(0, Num_Max)  # Se genera el número a modificar
                aux3 = np.random.randint(0, len(individuo[aux1]))  # Se genera qué índice de SD se va a modificar
                ind_asig = [i for i, v in enumerate(asig[aux1]) if v == individuo[aux1][aux3]]
                for j in ind_asig:
                    asig[aux1][j] = aux2    #Actualizamos la asignación de clases de la base mutada
                if individuo[aux1][aux3] == aux2:   #Hacemos esto para evitar repeticiones de SD en las listas de la solución
                    continue
                else:
                    individuo[aux1][aux3] = aux2
                    break
        else:
            aux2 = np.random.randint(0,Num_Max)                                   # Se genera el número a modificar
            ind_asig = [i for i,v in enumerate(asig[aux1]) if v != numero_supply_depots]
            for j in ind_asig:
                asig[aux1][j] = aux2    #Actualizamos la asignación de clases de la base mutada
            individuo[aux1] = aux2
        return individuo, asig
    def Comprobacion_Individuo (self, individuo, asig, capacidades):
        suma_comprobar = [[0 for _ in range(numero_clases)] for _ in range(self.Num_Max)]
        for i in range(self.Num_Max):   #Bucle para cada SD
            indices_bases = []
            for j, value in enumerate(individuo):
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
            for j in range(len(indices_bases)): #Bucle para base asociada a ese SD
                ind_clases = [k for k,v in enumerate(lista_clases_base[indices_bases[j]]) if isinstance(v, str)]    #Sacamos qué clases tiene esa base
                for k in ind_clases:    #Bucle para cada clase de esa base
                    if asig[indices_bases[j]][k] == i:    #Comprobamos que para esa clase esté asignado el SD correspondiente
                        comprobar_capacidades = capacidades[indices_bases[j]][k]
                        suma_comprobar[i][k] += comprobar_capacidades
            Caps_Clase_Comprobar = [t for t, value in enumerate(suma_comprobar[i]) if value > 200/numero_clases]
            if len(Caps_Clase_Comprobar) > 0:   #Si para un SD, se supera el umbral de al menos una clase... -> Reparación
                return True
        Caps_Comprobar = [ind_cap for ind_cap, j in enumerate(suma_comprobar) if sum(j) > 200]
        if len(Caps_Comprobar) > 0: #Si la suma de las capacidades pequeñas excede el límite de 200... -> Reparación
            return True
    def Reparacion_Aleatorio (self, individuo, capacidades): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
        capacidades_sd = list(np.zeros(self.Num_Max))  # Capacidades de los SD
        suma_capacidades = [[0 for _ in range(numero_clases)] for _ in range(self.Num_Max)]
        for i in range(self.Num_Max):
            indices_bases_reparar = [j for j, value in enumerate(individuo) if value == i]
            capacidades_sd_i = capacidades[indices_bases_reparar]
            capacidades_sd[i] = capacidades_sd_i
            suma_capacidades[i] = sum(capacidades_sd[i])  # Almacenamos todas las sumas de las capacidades en un array
        Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if sum(j) > 200]  # Comprobamos en qué SD's se han superado la capacidad
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
                Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if sum(j) > 200]  # Comprobamos en qué SD's se han superado la capacidad
                if len(Caps_SD_Superadas) == 0:
                    break
        return individuo

    def Reparacion_Mayor_Menor (self, individuo, asig, capacidades): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
        capacidades_sd = [[0 for _ in range(numero_clases)] for _ in range(self.Num_Max)]
        suma_capacidades = [[0 for _ in range(numero_clases)] for _ in range(self.Num_Max)]
        SD_ind = 0
        while True:
            indices_bases_reparar = []
            for j, value in enumerate(individuo):   #Sacamos bases asociadas a un SD
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
                    if asig[indices_bases_reparar[j]][k] == SD_ind:  # Comprobamos que para esa clase esté asignado el SD correspondiente
                        capacidades_sd_i = capacidades[indices_bases_reparar[j]][k]
                        capacidades_sd[SD_ind][k] = capacidades_sd_i
                        suma_capacidades[SD_ind][k] += capacidades_sd[SD_ind][k]    #Almacenamos todas las sumas de las capacidades en un array
            Caps_Clase_Comprobar = [t for t, value in enumerate(suma_capacidades[SD_ind]) if value > 200 / numero_clases]
            if len(Caps_Clase_Comprobar) > 0:  # Si para un SD, se supera el umbral de al menos una clase... -> Reparación
                while True: #Bucle para solucionar los SD
                    k_2 = np.argsort(suma_capacidades[SD_ind])[::-1]
                    k_1 = k_2[0]  # Solucionamos aquella capacidad que sea mas grande de las clases
                    while True: #Bucle infinito hasta que no dé errores para UNA CLASE DEL SD
                        while True:
                            SDs = list(np.arange(numero_supply_depots))  #Sacamos lista de SDs
                            while True:
                                k_3 = random.choice(SDs)    #Elegimos uno al azar
                                if k_3 == SD_ind:
                                    continue
                                else:
                                    break
                            indices_bases_SD = []   #Bases asociadas al SD de la Cap más grande
                            indices_resto_bases = []    #Bases asociadas al otro SD
                            for j, value in enumerate(individuo):  # Sacamos bases asociadas a un SD
                                if isinstance(value, list):
                                    # Si el elemento es una sublista, verificamos si tiene el SD
                                    if SD_ind in value:  #Tiene el SD asociado en el tipo de recurso que está dando problemas
                                        if asig[j][k_1] == SD_ind:
                                            indices_bases_SD.append(j)
                                    if k_3 in value:
                                        if asig[j][k_1] == k_3:
                                            indices_resto_bases.append(j)
                                else:
                                    # Si el elemento no es una sublista, verificamos si tiene el SD
                                    if value == SD_ind:
                                        if asig[j][k_1] == SD_ind:
                                            indices_bases_SD.append(j)
                                    if value == k_3:
                                        if asig[j][k_1] == k_3:
                                            indices_resto_bases.append(j)
                            capacidades_bases_SD_ordenados = list(np.argsort([capacidades[i][k_1] for i in indices_bases_SD])[::-1])
                            #Ordenamos índices de capacidades de esa clase de mayor a menor
                            indices_bases_SD_ordenados = [indices_bases_SD[i] for i in capacidades_bases_SD_ordenados]
                            #Ordenamos los índices de las bases al SD de la Cap más grande de mayor a menor
                            indice_base_1 = indices_bases_SD_ordenados[0]   #Cogemos la base con mayor cap de esa clase

                            lista_filtrada = [v for v in indices_resto_bases if capacidades[v][k_1] <= capacidades[indice_base_1][k_1] and capacidades[v][k_1] != 0]
                            if lista_filtrada:
                                indice_base_aleatoria_2 = random.choice(lista_filtrada)  # Elección aleatoria de la base del resto de bases
                            else:
                                SD_mismos_recursos = [v for i, v in enumerate(SDs) if v != SD_ind and v != k_3]    #Sacamos lista del resto de SDs
                                lista_ind = []
                                while True:  # Cotejamos que haya bases de la misma clase en otros SD
                                    while True:
                                        k_4 = random.choice(SD_mismos_recursos)  # Jugamos con uno de los SD con mismos recursos que las bases
                                        if k_4 == SD_ind:
                                            continue
                                        else:
                                            break
                                    indices_resto_bases = []  # Bases asociadas al otro SD
                                    for j, value in enumerate(individuo):  # Sacamos bases asociadas a un SD
                                        if isinstance(value, list):
                                            # Si el elemento es una sublista, verificamos si tiene el SD
                                            if k_4 in value and k_4 not in lista_ind:
                                                if asig[j][k_1] == k_4:
                                                    indices_resto_bases.append(j)
                                        else:
                                            # Si el elemento no es una sublista, verificamos si tiene el SD
                                            if value == k_4 and value not in lista_ind:
                                                if asig[j][k_1] == k_4:
                                                    indices_resto_bases.append(j)
                                    lista_filtrada = [v for v in indices_resto_bases if  capacidades[v][k_1] <= capacidades[indice_base_1][k_1] and capacidades[v][k_1] != 0]
                                    if lista_filtrada:
                                        k_3 = k_4   #Actualizamos el SD a reemplazar con otro aleatorio si el primero (el k_3) no funciona
                                        break
                                    else:  # Si no hay, la añadimos a una lista
                                        lista_ind.append(k_4)
                                        if len(lista_ind) == len(SD_mismos_recursos):  # Cuando el tamaño de lista sea igual que SD_mismos_recursos...
                                            e = random.randint(0, 5)
                                            f = indices_bases_SD_ordenados[0:e]
                                            for i in f:
                                                if isinstance(individuo[i], list):  #Si es una sublista, buscamos índice donde esté el valor de ese SD
                                                    indice = [j for j,v in enumerate(individuo[i]) if v == SD_ind]
                                                    if len(indice) == 1:
                                                        individuo[i][indice[0]] = k_3  #Lo cambiamos por el otro SD
                                                        asig[i][indice[0]] = k_3
                                                    else:
                                                        indi = random.choice(indice)
                                                        individuo[i][indi] = k_3  # Lo cambiamos por el otro SD
                                                        asig[i][indi] = k_3
                                                else:
                                                    individuo[i] = k_3  # ... Descargamos algunas bases del SD que nos da problemas sobre el otro (k_3)
                                                    indice_asig = [j for j, v in enumerate(asig[i]) if v != numero_supply_depots]
                                                    if len(indice_asig) == 1:
                                                        asig[i][indice_asig[0]] = k_3
                                                    elif len(indice_asig) > 1:
                                                        g = random.choice(indice_asig)
                                                        asig[i][g] = k_3
                                            continue
                                        else:
                                            continue
                                indice_base_aleatoria_2 = random.choice(lista_filtrada)

                            #Comprobamos si son listas de SD los índices cogidos
                            #Si lo son alguno (o los dos) -> Comprobamos que no tengan el SD que vaya a ponerle el otro
                            if isinstance(individuo[indice_base_1], list) or isinstance(individuo[indice_base_aleatoria_2], list):
                                if isinstance(individuo[indice_base_1], list) and isinstance(individuo[indice_base_aleatoria_2], list):
                                    p = [x for x, v in enumerate(individuo[indice_base_1]) if v == k_3]
                                    q = [x for x, v in enumerate(individuo[indice_base_aleatoria_2]) if v == SD_ind]
                                    if len(p) == 0 and len(q) == 0:
                                        break
                                elif isinstance(individuo[indice_base_1], list):
                                    p = [x for x, v in enumerate(individuo[indice_base_1]) if v == k_3]
                                    if len(p) == 0:
                                        break
                                elif isinstance(individuo[indice_base_aleatoria_2], list):
                                    q = [x for x, v in enumerate(individuo[indice_base_aleatoria_2]) if v == SD_ind]
                                    if len(q) == 0:
                                        break
                            else:
                                break

                        #Hacemos el intercambio de SD según el tipo de dato que tenga la base: int o list

                        if isinstance(individuo[indice_base_1], (int, np.integer)) and isinstance(individuo[indice_base_aleatoria_2], (int, np.integer)):
                            comprob_asig_1 = [i for i,v in enumerate(asig[indice_base_1]) if v != numero_supply_depots]
                            comprob_asig_2 = [i for i, v in enumerate(asig[indice_base_aleatoria_2]) if v != numero_supply_depots]
                            if len(comprob_asig_1) == 1 and len(comprob_asig_2) == 1:   #Si sólo tienen 1 clase las 2 bases... Las podemos intercambiar
                                if comprob_asig_1 == k_1 and comprob_asig_2 == k_1:
                                    asig[indice_base_1][comprob_asig_1], asig[indice_base_aleatoria_2][comprob_asig_2] = asig[indice_base_aleatoria_2][comprob_asig_2], asig[indice_base_1][comprob_asig_1]
                                    individuo[indice_base_1], individuo[indice_base_aleatoria_2] = individuo[indice_base_aleatoria_2], individuo[indice_base_1]
                            else:   #Si no, añadimos el SD a la lista de SDs de esa base
                                if len(comprob_asig_1) > 1:
                                    if k_1 in comprob_asig_1:
                                        asig[indice_base_1][k_1] = individuo[indice_base_aleatoria_2]  #Actualizamos la asignación de esa clase al nuevo SD
                                    else:
                                        asig[indice_base_1][random.choice(comprob_asig_1)] = individuo[indice_base_aleatoria_2]
                                    individuo[indice_base_1] = [individuo[indice_base_1]]   #Lo pasamos a lista
                                    individuo[indice_base_1].append(individuo[indice_base_aleatoria_2])    #Le añadimos el nuevo SD
                                    if k_1 in comprob_asig_2:
                                        if len(comprob_asig_2) > 1:
                                            asig[indice_base_aleatoria_2][k_1] = SD_ind  # Actualizamos la asignación de esa clase al viejo SD
                                            individuo[indice_base_aleatoria_2] = [individuo[indice_base_aleatoria_2]]  # Lo pasamos a lista
                                            individuo[indice_base_aleatoria_2].append(SD_ind)  # Le añadimos el viejo SD
                                        elif len(comprob_asig_2) == 1:
                                            if comprob_asig_2 == k_1:
                                                asig[indice_base_aleatoria_2][k_1] = SD_ind  # Actualizamos la asignación de esa clase al nuevo SD
                                                individuo[indice_base_aleatoria_2] = SD_ind

                                elif len(comprob_asig_1) == 1:
                                    if asig[indice_base_1][k_1] == asig[indice_base_1][comprob_asig_1] and asig[indice_base_1][k_1] != numero_supply_depots:
                                        asig[indice_base_1][k_1] = individuo[indice_base_aleatoria_2]  #Actualizamos la asignación de esa clase al nuevo SD
                                    else:
                                        asig[indice_base_1][comprob_asig_1] = individuo[indice_base_aleatoria_2]
                                    SD_aux = individuo[indice_base_1]
                                    individuo[indice_base_1] = individuo[indice_base_aleatoria_2]
                                    if k_1 in comprob_asig_2:
                                        if len(comprob_asig_2) > 1:
                                            asig[indice_base_aleatoria_2][k_1] = SD_aux # Actualizamos la asignación de esa clase al viejo SD
                                            individuo[indice_base_aleatoria_2] = [individuo[indice_base_aleatoria_2]]  # Lo pasamos a lista
                                            individuo[indice_base_aleatoria_2].append(SD_aux)  # Le añadimos el viejo SD
                                        elif len(comprob_asig_2) == 1:
                                            if comprob_asig_2 == k_1:
                                                asig[indice_base_aleatoria_2][k_1] = SD_aux # Actualizamos la asignación de esa clase al nuevo SD
                                                individuo[indice_base_aleatoria_2] = SD_aux

                                #Para hacer un intercambio equivalente, tenemos que comprobar que las 2 bases compartan la misma clase k

                                #FALTA HACER LA PRUEBA -> SI NOS DA PROBLEMAS, TENEMOS QUE COMPROBAR A LA HORA DE ELEGIR LA BASE ALEATORIA A INTERCAMBIAR
                                #QUE TENGA LA CLASE K EN ELLA

                        elif isinstance(individuo[indice_base_1], list) and isinstance(individuo[indice_base_aleatoria_2], list):
                            if asig[indice_base_1][k_1] != numero_supply_depots:
                                asig[indice_base_1][k_1] = k_3  # Actualizamos la asignación de esa clase al nuevo SD
                                comprob_asig_1 = [i for i, v in enumerate(asig[indice_base_1]) if v == SD_ind]
                                # Si al actualizar la asignación sigue estando el SD que estamos analizando -> No hacemos nada
                                # Sin embargo, si ya no aparece, tenemos que quitarlo de la lista
                                if len(comprob_asig_1) == 0 and SD_ind in individuo[indice_base_1]:
                                    ind_cambio = individuo[indice_base_1].index(SD_ind)
                                    individuo[indice_base_1].pop(ind_cambio)  # Le quitamos ese SD
                                individuo[indice_base_1].append(k_3)  # Le añadimos el nuevo SD SIEMPRE

                            if asig[indice_base_aleatoria_2][k_1] != numero_supply_depots:
                                asig[indice_base_aleatoria_2][k_1] = SD_ind  # Actualizamos la asignación de esa clase al nuevo SD
                                comprob_asig_2 = [i for i, v in enumerate(asig[indice_base_aleatoria_2]) if v == k_3]
                                # Si al actualizar la asignación sigue estando el SD que estamos analizando -> No hacemos nada
                                # Sin embargo, si ya no aparece, tenemos que quitarlo de la lista
                                if len(comprob_asig_2) == 0 and k_3 in individuo[indice_base_aleatoria_2]:
                                    ind_cambio = individuo[indice_base_aleatoria_2].index(k_3)
                                    individuo[indice_base_aleatoria_2].pop(ind_cambio)  # Le quitamos ese SD
                                individuo[indice_base_aleatoria_2].append(SD_ind)  # Le añadimos el nuevo SD SIEMPRE

                        elif isinstance(individuo[indice_base_1], list) or isinstance(individuo[indice_base_aleatoria_2], list):
                            if isinstance(individuo[indice_base_1], list):
                                if asig[indice_base_1][k_1] != numero_supply_depots:
                                    asig[indice_base_1][k_1] = individuo[indice_base_aleatoria_2]
                                    comprob_asig_1 = [i for i, v in enumerate(asig[indice_base_1]) if v == SD_ind]
                                    if len(comprob_asig_1) == 0 and SD_ind in individuo[indice_base_1]:
                                        ind_cambio = individuo[indice_base_1].index(SD_ind)
                                        individuo[indice_base_1].pop(ind_cambio)  # Le quitamos ese SD
                                    individuo[indice_base_1].append(individuo[indice_base_aleatoria_2])

                                if asig[indice_base_aleatoria_2][k_1] != numero_supply_depots:
                                    asig[indice_base_aleatoria_2][k_1] = SD_ind
                                    comprob_asig_1 = [i for i, v in enumerate(asig[indice_base_aleatoria_2]) if v != numero_supply_depots]
                                    if len(comprob_asig_1) == 1:
                                        individuo[indice_base_aleatoria_2] = SD_ind
                                    elif len(comprob_asig_1) > 1:
                                        individuo[indice_base_aleatoria_2] = [individuo[indice_base_aleatoria_2]]
                                        individuo[indice_base_aleatoria_2].append(SD_ind)
                            elif isinstance(individuo[indice_base_aleatoria_2], list):
                                if asig[indice_base_aleatoria_2][k_1] != numero_supply_depots:
                                    asig[indice_base_aleatoria_2][k_1] = individuo[indice_base_1]
                                    comprob_asig_2 = [i for i, v in enumerate(asig[indice_base_aleatoria_2]) if v == k_3]
                                    if len(comprob_asig_2) == 0 and k_3 in individuo[indice_base_aleatoria_2]:
                                        ind_cambio = individuo[indice_base_aleatoria_2].index(k_3)
                                        individuo[indice_base_aleatoria_2].pop(ind_cambio)  # Le quitamos ese SD
                                    individuo[indice_base_aleatoria_2].append(individuo[indice_base_1])

                                if asig[indice_base_1][k_1] != numero_supply_depots:
                                    asig[indice_base_1][k_1] = k_3
                                    comprob_asig_2 = [i for i, v in enumerate(asig[indice_base_1]) if v != numero_supply_depots]
                                    if len(comprob_asig_2) == 1:
                                        individuo[indice_base_1] = k_3
                                    elif len(comprob_asig_2) > 1:
                                        individuo[indice_base_1] = [individuo[indice_base_1]]
                                        individuo[indice_base_1].append(k_3)

                        indices_bases_SD = []  # Bases asociadas al SD de la Cap más grande
                        for j, value in enumerate(individuo):  # Sacamos bases asociadas a un SD
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
                                if asig[indices_bases_SD[j]][t] == SD_ind:  # Comprobamos que para esa clase esté asignado el SD correspondiente
                                    capacidades_sd_i = capacidades[indices_bases_SD[j]][t]
                                    capacidades_sd[SD_ind][t] = capacidades_sd_i
                                    suma_capacidades[SD_ind][t] += capacidades_sd[SD_ind][t]  # Almacenamos todas las sumas de las capacidades en un array
                        if suma_capacidades[SD_ind][k_1] > 200/numero_clases:
                            continue
                        else:
                            break
                    Caps_Clase_Comprobar = [t for t, value in enumerate(suma_capacidades[SD_ind]) if value > 200 / numero_clases]
                    if len(Caps_Clase_Comprobar) == 0:  #Comprobamos las clases para el SD -> Si no superan el umbral, seguimos
                        break
                Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if sum(j) > 200]  # Comprobamos en qué SD's se han superado la capacidad
                if len(Caps_SD_Superadas) > 0:
                    SD_ind = Caps_SD_Superadas[0]   #Elegimos el primer SD que nos salga de la lista
                else:
                    SD_ind = 0  #Si no hay ninguno, hacemos un barrido por todos los SD
            else:
                SD_ind += 1
                if SD_ind == numero_supply_depots:
                    Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if sum(j) > 200]  # Comprobamos en qué SD's se han superado la capacidad
                    if len(Caps_SD_Superadas) > 0:
                        SD_ind = 0
                    else:
                        break
        return individuo, asig

def Puntos_Sin_Repetir(num_points, offset=0.5):
    points = set()  # Usamos un conjunto para evitar duplicados
    mapa_dem = 'PNOA_MDT05_ETRS89_HU30_0560_LID.tif'
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
            if isinstance(SD, list):    #Si hay muchos SD asociados a una base
                dist_SD = []
                for k in SD:   #Obtenemos todas las distancias de las bases a los SD y obtenemos la mínima
                    dist_SD.append(distancias[k][j])
                min_dist = min(dist_SD)
                fitness += min_dist
            else:
                fitness += distancias[SD][j]    #Calculo fitness buscando en la matriz de distancias la distancia asociada

        fitness = fitness/numero_bases
        lista_fitness.append(fitness)
    return lista_fitness

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

    Pob_Actual = []
    Costes = []
    numero_bases = 200
    numero_supply_depots = 30
    capacidad_maxima = 20
    numero_clases = 5
    Ruta_Puntos = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Bases_SD_1.csv")
    Ruta_Capacidades = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Cap_Bases_SD_1.csv")
    Ruta_Clases_Bases = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Clases_Bases_1.csv")
    Ruta_Caps_Clases_Bases = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Caps_Clases_Bases_1.csv")
    Ruta_Caps_Clases_SD = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Tipos_Recursos_B',
        f"Caps_Clases_SD_1.csv")
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
    latitudes_bases, longitudes_bases = zip(*bases)
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
    latitudes_supply_depots, longitudes_supply_depots = zip(*supply_depots)
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

    #clases = np.random.randint(0, 2, size=(Tam_Individuos , numero_clases))  #Ponemos a 0 ó a 1 las clases de recursos en cada -> A 1 las que

    distancias_euclideas = Distancia_Base_Supply_Depot_2D(bases, supply_depots) #Obtenemos distancias de bases a supply depots
    #distancias_euclideas_orden = []
    #for j in indices_capacidad_bases:
    #    distancias_euclideas_aux = []
    #    for i in range(0, numero_supply_depots):
    #        distancias_euclideas_aux.append(distancias_euclideas[i][j])
    #    distancias_euclideas_orden.append(distancias_euclideas_aux) #Distancias de una base hacia los supply depot ordenadas según su capacidad
    #distancias_capacidades_bases = list(zip(*[capacidad_bases[indices_capacidad_bases]], [distancias_euclideas_orden[l] for l in range(0, 200)])) #Ordenamos en una única lista

    ### A CONTINUACIÓN, APLICAMOS EL ALGORITMO DESPUÉS DE OBTENER LOS COSTES Y DISTANCIAS
    
    Ev1 = EvolutiveClass(Num_Individuos, Num_Generaciones, Tam_Individuos,numero_supply_depots, Prob_Padres, Prob_Mutacion, Prob_Cruce)
    Costes_Generacion = []
    Lista_Asig_Clases_SD = []
    Pob_Inicial, Lista_Asig_Clases_SD = Ev1.PoblacionInicial(capacidad_bases, Lista_Asig_Clases_SD, 100, numero_bases, numero_supply_depots,)  #Poblacion inicial -> 100 posibles soluciones -> PADRES
    for i in range(Num_Generaciones):
        print(("Generación: " + str(i + 1)))
        Fitness = Funcion_Fitness(distancias_euclideas, Pob_Inicial)
        Pob_Actual, Lista_Asig_Clases_SD, Costes = Ev1.Seleccion(Pob_Inicial,Lista_Asig_Clases_SD, Fitness)
        Pob_Inicial, Lista_Asig_Clases_SD = Ev1.Cruce(Pob_Actual,Lista_Asig_Clases_SD, capacidad_bases, numero_supply_depots)   #Aplicamos cruce en las soluciones
        print("Coste: " + str(Costes[0]))
        Costes_Generacion.append(Costes[0])
    Sol_Final = Pob_Inicial[0]   #La primera población será la que tenga menor coste
    Coste_Final = Costes[0]
    print("Solución final:")
    for j in range(Tam_Individuos):
        print("Base " + str(j) + "-> SD: " + str(Sol_Final[j]))
    print("Coste de la solución: " + str(Coste_Final))


    # Graficar el mapa y los puntos
    fig_1 = plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', s=60,label='Puntos de Suministro')
    plt.gca().invert_yaxis()
    fig_1.show()
    #Evolución del coste de una de las rutas
    coste = plt.figure(figsize=(10, 6))
    plt.plot(Costes_Generacion)
    coste.show()
    #Graficar solución
    colores = ['green', 'blue', 'red', 'orange', 'purple']  #Lista de colores para cada tipo de recurso
    fig = plt.figure(figsize=(10, 6))
    ejes = fig.add_subplot(111) #Creamos ejes en la figura (1 fila, 1 columna y 1 cuadrícula) -> Necesarios para dibujar los puntos multicolor
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', s=60, label='Puntos de Suministro')
    for k in range(numero_supply_depots):
        SD = []
        for j, value in enumerate(Sol_Final):   #Obtenemos lista de índices de las bases de la solución que tienen el SD asociado
            if isinstance(value, list):
                # Si el elemento es una sublista, verificamos si tiene el SD
                if k in value:
                    SD.append(j)
            else:
                # Si el elemento no es una sublista, verificamos si tiene el SD
                if value == k:
                    SD.append(j)
        for t in SD:    #Bucle para pintar puntos multicolor
            clases_recurso = [i for i, v in enumerate(lista_clases_base[t]) if isinstance(v, str)]  # Sacamos índices de la lista de clases de la base
            lista_colores = [colores[j] for j, x in enumerate(clases_recurso)]  # Obtenemos los colores para cada clase dentro del punto que dibujaremos
            angulo_color = 360/len(lista_colores)
            for s, color in enumerate(lista_colores):
                angulo_inicio = s*angulo_color
                seccion_circulo = patches.Wedge((longitudes_bases[t],latitudes_bases[t]),50,angulo_inicio,angulo_inicio+angulo_color, color=color)
                ejes.add_patch(seccion_circulo)
        #ejes.set_xlim(-5, 185)  # Limitar el eje x de 0 a 180
        #ejes.set_ylim(-5, 185)  # Limitar el eje y de 0 a 180
        ejes.set_aspect('equal')    #Para que los puntos multicolor no queden ovalados, sino circulares
        if len(SD) > 0: #Porque puede haber bases que no tengan asociado el SD de la iteración que toca
            aux = random.choice(SD) #Punto del que saldrán las líneas a los SD
            if isinstance(Sol_Final[aux], list):  #Si un punto cualquiera está asociado a más de un SD
                for i in range(len(Sol_Final[aux])):
                    plt.plot([longitudes_bases[aux], longitudes_supply_depots[Sol_Final[aux][i]]],[latitudes_bases[aux], latitudes_supply_depots[Sol_Final[aux][i]]], color='red')
            else:
                plt.plot([longitudes_bases[aux], longitudes_supply_depots[Sol_Final[aux]]],[latitudes_bases[aux], latitudes_supply_depots[Sol_Final[aux]]], color='red')
        else:
            continue
    plt.xlabel('Distancia Horizontal (px/m)')
    plt.ylabel('Distancia Vertical (px/m)')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.gca().invert_yaxis()
    plt.show()
