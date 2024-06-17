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
        Pob_Ini_List = []
        for i in range(Fil):
            Pob_Ini = np.random.randint(0,Num_Max, size=(1,Col))  #Son los índices de los SD asignados a cada base
            x = np.full(Col, 200)
            Pob_Ini = np.vstack((Pob_Ini,x))
            if(self.Comprobacion_Individuo(Pob_Ini, Capacidades, distancias_euclideas)):
                Pob_Ini = self.Reparacion_Mayor_Menor(Pob_Ini, Capacidades, distancias_euclideas)
                Pob_Ini_List.append(Pob_Ini)
            else:
                Pob_Ini_List.append(Pob_Ini)
        return Pob_Ini_List

    def Seleccion(self, poblacion_inicial, coste):
        index = np.argsort(coste)
        coste_ordenado = np.sort(coste)
        poblacion_actual = poblacion_inicial[index][:]   #La población tendrá más soluciones que la inicial debido al cruce
        poblacion_actual = poblacion_actual[0:self.Num_Individuos][:]    #Nos quedamos con los mejores individuos
        return poblacion_actual, coste_ordenado

    def Cruce (self, poblacion, capacidades, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        capacidades_sd = list(np.zeros(self.Num_Max))   #Capacidades de los SD
        #Indice_Seleccionado = []
        Indices_Validos = list(np.arange(self.Num_Individuos))

        for v in range(self.Num_Individuos - self.Num_Padres): #Bucle para generar HIJOS
            Indice_Padres = random.sample(Indices_Validos, 2)
            #Indice_Padres = random.sample([j for j in Indices_Validos if j not in Indice_Seleccionado], 2)            # Se elige aleatoriamente el indice de los padres
            #Indice_Seleccionado.extend(Indice_Padres)   #Guardamos los índices elegidos para que no los vuelva a repetir en la siguiente iteración
            Padre1 = poblacion[Indice_Padres[0]][:]                              # Se coge el padre 1
            Padre2 = poblacion[Indice_Padres[1]][:]                              # Se coge el padre 2
            Hijo = np.copy(Padre1)                                              # El hijo va a ser una copia del padre 1
            vector = 1*(np.random.rand(self.Tam_Individuos) > self.Prob_Cruce)  # Se genera un vector para seleccionar los genes del padre 2
            Hijo[0][np.where(vector==1)[0]] = Padre2[0][np.where(vector==1)[0]]       # Los genes seleccionados del padre 2 pasan al hijo
            if np.random.rand() < self.Prob_Mutacion:                           # Se comprueba si el hijo va a mutar
                Hijo = self.Mutacion(Hijo, Num_Max)
            Hijo[1, :] = numero_bases   #Limpiamos la fila de intermediarios

            #TRAS MUTAR, SACAMOS LA LISTA DE BASES A INTERMEDIARIOS DE ESE HIJO -> LA AÑADIMOS AL RESTO

            for i in range(self.Num_Max):
                capacidades_sd_i = list(np.zeros(numero_bases))  # Array auxiliar del tamaño del número de bases para almacenar las capacidades
                #indices_bases_inter = list(np.zeros(numero_bases))  # Lista de índices de bases a intermediarios
                indices_bases = np.where(Hijo[0][ind_bases_antes] == i)[0]  # Obtenemos los indices de las bases asociadas a un SD "i"
                for ind in ind_bases_antes[indices_bases]:
                    capacidades_sd_i[ind] = capacidades[ind]  # Copiamos las capacidades originales en otra variable auxiliar
                ind_inter = np.where(Hijo[0][ind_intermediarios] == i)[0]  # Buscamos qué intermediarios tienen ese SD asociado
                for ind in ind_intermediarios[ind_inter]:
                    capacidades_sd_i[ind] += capacidades[ind]  # Copiamos las capacidades originales en otra variable auxiliar
                capacidades_sd[i] = capacidades_sd_i

                #CAPACIDADES COPIADAS

                for k in ind_intermediarios[ind_inter]:  # Comprobamos para esos intermediarios sus distancias con el SD y con las bases cercanas
                    contador = 0  # Con este contador vamos sumando las capacidades de las bases que coteja el intermediario
                    indices_bases_inter = list(np.full(numero_bases,numero_bases))  # Lista de índices de bases a intermediarios
                    for j in ind_bases_antes[indices_bases]:
                        distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[k])  # Distancia entre una base y el intermediario
                        if distancia_base_inter < distancias_euclideas[i][j] and capacidades[j] <= (capacidades[k] - contador):
                            # Si esa distancia es menor que la de la base al SD
                            # y la capacidad de la base es menor o igual que la diferencia entre la cap del intermediario y la suma de cap de las bases que contempla...
                            capacidades_sd_i[j] = 0  # No contemplamos la capacidad de esa base -> Va englobada en la capacidad del intermediario
                            contador += capacidades[j]  # Sumamos esa capacidad al contador y vemos la siguiente base
                            indices_bases_inter[j] = k  # Guardamos el valor del intermediario en la posición de la base
                        if indices_bases_inter[j] in ind_intermediarios:
                            Hijo[1][j] = indices_bases_inter[j]  # Añadimos en el indice que corresponde con el número del SD los índices recogidos
                    for s in range(numero_bases):
                        if Hijo[1][s] not in ind_intermediarios:
                            Hijo[1][s] = numero_bases  # Nos aseguramos de que el valor que no esté dentro de ind_intermediarios no interfiera

            # SACAMOS LA LISTA PARA ESE HIJO

            if(self.Comprobacion_Individuo(Hijo, capacidades, distancias_euclideas)):                    # Se comprueba si hay que reparar el hijo
                 Hijo = self.Reparacion_Mayor_Menor(Hijo, capacidades, distancias_euclideas)
            poblacion = np.insert(poblacion,self.Num_Individuos+v,Hijo, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado
        return poblacion

    def Mutacion (self, individuo, Num_Max=None):                                
        aux1 = np.random.randint(0, individuo[0].shape[0])                         # Se genera número aleatorio para ver la posición que muta
        aux2 = np.random.randint(0,Num_Max)                                   # Se genera el número a modificar
        individuo[0][aux1] = aux2
        return individuo
    def Comprobacion_Individuo (self, individuo, capacidades, distancias):
        suma_comprobar = list(np.zeros(self.Num_Max))
        for i in range(self.Num_Max):
            comprobar_capacidades = list(np.zeros(numero_bases))  # Array auxiliar del tamaño del número de bases para almacenar las capacidades
            indices_bases = np.where(individuo[0][ind_bases_antes] == i)[0]  #Obtenemos los indices de las bases asociadas a un SD "i"
            for ind in ind_bases_antes[indices_bases]:
                comprobar_capacidades[ind] = capacidades[ind]   #Copiamos las capacidades originales en otra variable auxiliar
            ind_inter = np.where(individuo[0][ind_intermediarios] == i)[0]  # Buscamos qué intermediarios tienen ese SD asociado
            for ind in ind_intermediarios[ind_inter]:
                comprobar_capacidades[ind] = capacidades[ind]   #Copiamos las capacidades originales en otra variable auxiliar

            #YA TENEMOS LAS CAPACIDADES COPIADAS -> PROCEDEMOS A LA COMPROBACIÓN

            for k in ind_intermediarios[ind_inter]:  # Comprobamos para esos intermediarios sus distancias con el SD y con las bases cercanas
                contador = 0    #Con este contador vamos sumando las capacidades de las bases que coteja el intermediario
                indices_bases_inter = list(np.full(numero_bases,numero_bases))  # Lista de índices de bases a intermediarios
                for j in ind_bases_antes[indices_bases]:
                    distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[k])  #Distancia entre una base y el intermediario
                    if distancia_base_inter < distancias[i][j] and capacidades[j] <= (capacidades[k] - contador):
                        # Si esa distancia es menor que la de la base al SD
                        # y la capacidad de la base es menor o igual que la diferencia entre la cap del intermediario y la suma de cap de las bases que contempla...
                        comprobar_capacidades[j] = 0    #No contemplamos la capacidad de esa base -> Va englobada en la capacidad del intermediario
                        contador += capacidades[j]  #Sumamos esa capacidad al contador y vemos la siguiente base
                        indices_bases_inter[j] = k
                    if indices_bases_inter[j] in ind_intermediarios:
                        individuo[1][j] = indices_bases_inter[j]  # Añadimos en el indice que corresponde con el número del SD los índices recogidos
                    #for s in range(numero_bases):
                        #if individuo[1][s] not in ind_intermediarios:
                            #individuo[1][s] = numero_bases  # Nos aseguramos de que el valor que no esté dentro de ind_intermediarios no interfiera
            individuo = individuo.astype(int)
            A = individuo[1][np.array([i for i, x in enumerate(individuo[1]) if x != 200], dtype=int)]  #Saco índices de bases asociadas a inters
            B = individuo[0][A]     #Miro el SD al que están asociadas
            C = individuo[0][np.array([i for i, x in enumerate(individuo[1]) if x != 200], dtype=int)]  #Saco los SD's asociados a los inters
            suma_comprobar[i] = sum(comprobar_capacidades)
            if not np.array_equal(B,C):    #Si B y C no son iguales -> Reparación -> Porque no puede haber bases asociadas a un inter que vayan a un SD distinto al del inter
                return True
        Caps_Comprobar = [ind_cap for ind_cap, j in enumerate(suma_comprobar) if j > 200]
        if len(Caps_Comprobar) > 0:
            return True

    def Reparacion_Aleatorio (self, individuo, capacidades, distancias): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
        capacidades_sd = list(np.zeros(self.Num_Max))   #Capacidades de los SD
        suma_capacidades = list(np.zeros(self.Num_Max)) #Suma de las capacidades de las bases
        capacidades_sd_i = list(np.zeros(numero_bases))  # Array auxiliar del tamaño del número de bases para almacenar las capacidades
        for i in range(self.Num_Max):
            indices_bases = np.where(individuo[ind_bases_antes] == i)[0]  #Obtenemos los indices de las bases asociadas a un SD "i"
            for ind in ind_bases_antes[indices_bases]:
                capacidades_sd_i[ind] = capacidades[ind]   #Copiamos las capacidades originales en otra variable auxiliar
            ind_inter = np.where(individuo[ind_intermediarios] == i)[0]  # Buscamos qué intermediarios tienen ese SD asociado
            for ind in ind_intermediarios[ind_inter]:
                capacidades_sd_i[ind] = capacidades[ind]   #Copiamos las capacidades originales en otra variable auxiliar
            capacidades_sd[i] = capacidades_sd_i

            #CAPACIDADES COPIADAS

            for k in ind_intermediarios[ind_inter]:  # Comprobamos para esos intermediarios sus distancias con el SD y con las bases cercanas
                contador = 0    #Con este contador vamos sumando las capacidades de las bases que coteja el intermediario
                for j in ind_bases_antes[indices_bases]:
                    distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[k])  #Distancia entre una base y el intermediario
                    if distancia_base_inter < distancias[i][j] and capacidades[j] <= (capacidades_sd_i[k] - contador):
                        # Si esa distancia es menor que la de la base al SD
                        # y la capacidad de la base es menor o igual que la diferencia entre la cap del intermediario y la suma de cap de las bases que contempla...
                        capacidades_sd_i[j] = 0    #No contemplamos la capacidad de esa base -> Va englobada en la capacidad del intermediario
                        contador += capacidades[j]  #Sumamos esa capacidad al contador y vemos la siguiente base
            suma_capacidades[i] = sum(capacidades_sd_i)  # Almacenamos todas las sumas de las capacidades en un array

        # YA TENEMOS LAS CAPACIDADES COPIADAS -> PROCEDEMOS A LA COMPROBACIÓN

        Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  #Comprobamos en qué SD's se han superado la capacidad
        for k in Caps_SD_Superadas:
            while True:
                indices_bases_inters_SD = [j for j, value in enumerate(individuo) if value == k]    #Obtenemos índices de las bases cuya suma de caps supera el umbral
                indices_resto_bases_inters = [j for j, value in enumerate(individuo) if value != k]  # Obtenemos índices del resto de bases
                indice_base_aleatoria_1 = random.choice(indices_bases_inters_SD) #Elección aleatoria de la base del SD

                indice_base_aleatoria_2 = random.choice(indices_resto_bases_inters)  # Elección aleatoria de la base del resto de bases
                individuo[indice_base_aleatoria_1], individuo[indice_base_aleatoria_2] = individuo[indice_base_aleatoria_2], individuo[indice_base_aleatoria_1] #Intercambio posiciones de las bases

                #indices_bases_reparadas = [j for j, value in enumerate(individuo) if value == k]    #Obtenemos índices de las bases cuya suma de caps supera el umbral

                indices_bases = np.where(individuo[ind_bases_antes] == k)[0]  # Obtenemos los indices de las bases asociadas a un SD "i"
                for ind in ind_bases_antes[indices_bases]:
                    capacidades_sd_i[ind] = capacidades[ind]  # Copiamos las capacidades originales en otra variable auxiliar
                ind_inter = np.where(individuo[ind_intermediarios] == k)[0]  # Buscamos qué intermediarios tienen ese SD asociado
                for ind in ind_intermediarios[ind_inter]:
                    capacidades_sd_i[ind] = capacidades[ind]  # Copiamos las capacidades originales en otra variable auxiliar
                capacidades_sd[k] = capacidades_sd_i

                #TENEMOS CAPACIDADES ACTUALIZADAS

                for i in ind_intermediarios[ind_inter]:  # Comprobamos para esos intermediarios sus distancias con el SD y con las bases cercanas
                    contador = 0  # Con este contador vamos sumando las capacidades de las bases que coteja el intermediario
                    for j in ind_bases_antes[indices_bases]:
                        distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[k])  # Distancia entre una base y el intermediario
                        if distancia_base_inter < distancias[k][j] and capacidades_sd_i[j] <= (capacidades_sd_i[i] - contador):
                            # Si esa distancia es menor que la de la base al SD
                            # y la capacidad de la base es menor o igual que la diferencia entre la cap del intermediario y la suma de cap de las bases que contempla...
                            capacidades_sd_i[j] = 0  # No contemplamos la capacidad de esa base -> Va englobada en la capacidad del intermediario
                            contador += capacidades[j]  # Sumamos esa capacidad al contador y vemos la siguiente base
                if sum(capacidades_sd_i) > 200: #Si es más de 200, volvemos a hacer mismo código del while
                    continue
                else: #Si no, salimos del while y avanzamos en el for
                    break
        return individuo

    def Reparacion_Mayor_Menor (self, individuo, capacidades, distancias): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
        capacidades_sd = list(np.zeros(self.Num_Max))  # Capacidades de los SD
        suma_capacidades = list(np.zeros(self.Num_Max))  # Suma de las capacidades de las bases
        individuo[1, :] = numero_bases  # Limpiamos la fila de intermediarios
        for i in range(self.Num_Max):
            capacidades_sd_i = list(np.zeros(numero_bases))  # Array auxiliar del tamaño del número de bases para almacenar las capacidades
            indices_bases = np.where(individuo[0][ind_bases_antes] == i)[0]  #Obtenemos los indices de las bases asociadas a un SD "i"
            for ind in ind_bases_antes[indices_bases]:
                capacidades_sd_i[ind] = capacidades[ind]   #Copiamos las capacidades originales en otra variable auxiliar
            ind_inter = np.where(individuo[0][ind_intermediarios] == i)[0]  # Buscamos qué intermediarios tienen ese SD asociado
            for ind in ind_intermediarios[ind_inter]:
                capacidades_sd_i[ind] += capacidades[ind]   #Copiamos las capacidades originales en otra variable auxiliar
            capacidades_sd[i] = capacidades_sd_i

            # CAPACIDADES COPIADAS

            for k in ind_intermediarios[ind_inter]:  # Comprobamos para esos intermediarios sus distancias con el SD y con las bases cercanas
                contador = 0  # Con este contador vamos sumando las capacidades de las bases que coteja el intermediario
                indices_bases_inter = list(np.full(numero_bases,numero_bases))  # Lista de índices de bases a intermediarios
                for j in ind_bases_antes[indices_bases]:
                    distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[k])  # Distancia entre una base y el intermediario
                    if distancia_base_inter < distancias[i][j] and capacidades[j] <= (capacidades[k] - contador):
                        # Si esa distancia es menor que la de la base al SD
                        # y la capacidad de la base es menor o igual que la diferencia entre la cap del intermediario y la suma de cap de las bases que contempla...
                        capacidades_sd_i[j] = 0  # No contemplamos la capacidad de esa base -> Va englobada en la capacidad del intermediario
                        contador += capacidades[j]  # Sumamos esa capacidad al contador y vemos la siguiente base
                        indices_bases_inter[j] = k  #Guardamos el valor del intermediario en la posición de la base
                    if indices_bases_inter[j] in ind_intermediarios:
                        individuo[1][j] = indices_bases_inter[j]  # Añadimos en el indice que corresponde con el número del SD los índices recogidos
                #for s in range(numero_bases):
                    #if individuo[1][s] not in ind_intermediarios:
                        #individuo[1][s] = numero_bases  # Nos aseguramos de que el valor que no esté dentro de ind_intermediarios no interfiera
            suma_capacidades[i] = sum(capacidades_sd_i)  # Almacenamos todas las sumas de las capacidades en un array

        # YA TENEMOS LAS CAPACIDADES COPIADAS -> PROCEDEMOS A LA COMPROBACIÓN

        Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  #Comprobamos en qué SD's se han superado la capacidad
        if len(Caps_SD_Superadas) > 0:
            while True:
                k_2 = np.argsort(suma_capacidades)[::-1]
                k = k_2[0]  # Solucionamos aquella capacidad que sea mas grande
                while True:
                    k_3 = random.choice(k_2[len(suma_capacidades) - 4:len(suma_capacidades)])      #Jugamos con uno de los 4 SD con menos suma de bases
                    indices_bases_inters_SD = [j for j, value in enumerate(individuo[0]) if value == k]    #Obtenemos índices de las bases cuya suma de caps supera el umbral
                    indices_resto_bases = [j for j, value in enumerate(individuo[0]) if value == k_3]  # Obtenemos índices del resto de bases
                    capacidades_bases_SD_ordenados = list(np.argsort([capacidades[i] for i in indices_bases_inters_SD])[::-1])
                    indices_bases_SD_ordenados = [indices_bases_inters_SD[i] for i in capacidades_bases_SD_ordenados]

                    indice_base_1 = indices_bases_SD_ordenados[0]  # Elegimos una de las 5 bases del SD con mayor capacidad
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
                        individuo[0][indice_base_1], individuo[0][indice_base_aleatoria_2] = individuo[0][indice_base_aleatoria_2],individuo[0][indice_base_1]  # Intercambio posiciones de las bases
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
                        capacidades_sd_i[ind] += capacidades[ind]  # Copiamos las capacidades originales en otra variable auxiliar
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
                        indices_bases_inter = list(np.full(numero_bases, numero_bases))  # Lista de índices de bases a intermediarios
                        for j in ind_bases_antes[indices_bases]:
                            distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[v])  # Distancia entre una base y el intermediario
                            if distancia_base_inter < distancias[k][j] and capacidades[j] <= (capacidades[v] - contador):
                                # Si esa distancia es menor que la de la base al SD
                                # y la capacidad de la base es menor o igual que la diferencia entre la cap del intermediario y la suma de cap de las bases que contempla...
                                capacidades_sd_i[j] = 0  # No contemplamos la capacidad de esa base -> Va englobada en la capacidad del intermediario
                                contador += capacidades[j]  # Sumamos esa capacidad al contador y vemos la siguiente base
                                indices_bases_inter[j] = v  # Guardamos el valor del intermediario en la posición de la base
                            if indices_bases_inter[j] in ind_intermediarios:
                                individuo[1][j] = indices_bases_inter[j]  # Añadimos en el indice que corresponde con el número del SD los índices recogidos
                        #for s in range(numero_bases):
                            #if individuo[1][s] not in ind_intermediarios:
                                #individuo[1][s] = numero_bases  # Nos aseguramos de que el valor que no esté dentro de ind_intermediarios no interfiera
                    for s in ind_intermediarios[ind_inter]:
                        counter = 0
                        for t in range(len(individuo[1])):
                            if individuo[1][t] == s:  # Si una base pertenece a un intermediario añadimos su capacidad
                                counter += capacidades[t]
                        if capacidades[s] < counter:  # Si la suma de caps de bases supera al intermediario, salimos de los bucles y se vuelve a hacer el while True
                            indices_bases_mas = np.array([i for i, value in enumerate(individuo[1]) if value == s])  # Sacamos índices de bases asociadas al intermediario
                            capacidades_orden = list(np.argsort([capacidades[i] for i in indices_bases_mas])[::-1])
                            individuo[1][indices_bases_mas[capacidades_orden[0]]] = 200  # La base que tenga más capacidad deja de pertenecer a ese intermediario -> Priorizamos las pequeñas
                            capacidades_sd_i[indices_bases_mas[capacidades_orden[0]]] = capacidades[indices_bases_mas[capacidades_orden[0]]]
                    if sum(capacidades_sd_i) > 200: #Si es más de 200, volvemos a hacer mismo código del while
                        continue
                    else: #Si no, salimos del while y avanzamos en el for
                        capacidades_sd[k] = capacidades_sd_i
                        suma_capacidades[k] = sum(capacidades_sd_i)
                        break

                # CUANDO SOLUCIONAMOS UNA SUMA, LA GUARDAMOS, Y SOLUCIONAMOS LA SIGUIENTE SIN ASEGURARNOS DE QUE LA ANTERIOR ESTÉ BIEN
                # TENEMOS QUE AÑADIR CÓDIGO PARA VER QUE NO PASE ESO Y QUE TODAS LAS SUMAS ESTÉN BIEN ACTUALIZADAS

                for i in range(numero_supply_depots):   #Bucle para comprobar sumas de capacidades alrededor de un SD
                    F = np.array([t for t, x in enumerate(individuo[0]) if x == i], dtype=int)
                    H = np.where(individuo[1][F] == 200)
                    I = np.array(capacidades)
                    suma_capacidades[i] = sum(I[F[H]])

                Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad
                if len(Caps_SD_Superadas) == 0:
                    break

        return individuo

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
def Funcion_Fitness(distancias, poblacion):
    lista_fitness = []
    for i in range(len(poblacion)):    #Aplicamos la función fitness a cada solución
        fitness = 0
        for j in range(len(poblacion[i][0])):   #Bucle para recorrer bases que no sean intermediarios
            SD_base = int(poblacion[i][0][j])    #Saco el SD asociado a una base de la población
            ind_inter = np.where(poblacion[i][0][ind_intermediarios] == SD_base)[0]   #Buscamos qué intermediarios tienen ese SD asociado
            for k in ind_inter: #Comprobamos para esos intermediarios sus distancias con la base elegida
                distancia_base_inter = Distancia_Base_Supply_Depot_2D(bases_inter[j], bases_inter[k])
                if distancia_base_inter < distancias[SD_base][j]:   #Si esa distancia es menor que la de la base al SD
                    fitness += distancia_base_inter  # Calculo fitness usando la distancia de la base al intermediario
                else:
                    continue
            fitness += distancias[SD_base][j]    #Calculo fitness buscando en la matriz de distancias la distancia asociada
        fitness = fitness/numero_bases
        lista_fitness.append(fitness)
    return lista_fitness

if __name__ == "__main__":
    # Definicion de los parámetros del genético
    Num_Individuos = 100
    Num_Generaciones = 100
    Tam_Individuos = 200
    Prob_Padres = 0.1
    Prob_Mutacion = 0.01
    Prob_Cruce = 0.5

    Pob_Actual = []
    Lista_Bases_Actual = []
    Costes = []
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
    #distancias_euclideas_orden = []
    #for j in indices_capacidad_bases:
    #    distancias_euclideas_aux = []
    #    for i in range(0, numero_supply_depots):
    #        distancias_euclideas_aux.append(distancias_euclideas[i][j])
    #    distancias_euclideas_orden.append(distancias_euclideas_aux) #Distancias de una base hacia los supply depot ordenadas según su capacidad
    #distancias_capacidades_bases = list(zip(*[capacidad_bases[indices_capacidad_bases]], [distancias_euclideas_orden[l] for l in range(0, 200)])) #Ordenamos en una única lista

    ### A CONTINUACIÓN, APLICAMOS EL ALGORITMO DESPUÉS DE OBTENER LOS COSTES Y DISTANCIAS

    for v in range(Tam_Individuos):
        if v in ind_intermediarios:
            print("Intermediario: " + str(v) + " -> Capacidad: " + str(capacidad_bases[v]))

    Ev1 = EvolutiveClass(Num_Individuos, Num_Generaciones, Tam_Individuos,numero_supply_depots, Prob_Padres, Prob_Mutacion, Prob_Cruce)
    #Ev1.ImprimirInformacion()
    Pob_Inicial = Ev1.PoblacionInicial(capacidad_bases, 100, numero_bases, numero_supply_depots)  #Poblacion inicial -> 100 posibles soluciones -> PADRES
    for i in range(Num_Generaciones):
        print(("Generación: " + str(i+1)))
        Pob_Actual = Ev1.Cruce(Pob_Inicial, capacidad_bases, numero_supply_depots)   #Aplicamos cruce en las soluciones
        Fitness = Funcion_Fitness(distancias_euclideas, Pob_Actual)
        Pob_Inicial, Costes = Ev1.Seleccion(Pob_Actual, Fitness)
        print("Coste: " + str(Costes[0]))
    Sol_Final = Pob_Inicial[0]   #El primer individuo de la población será el que tenga menor coste
    Coste_Final = Costes[0]
    print("Solución final:")
    for j in range(Tam_Individuos):
        print("Base " + str(j) + "-> SD: " + str(Sol_Final[0][j]) + " -> Intermediario: " + str(Sol_Final[1][j]) + " -> Capacidad: " + str(capacidad_bases[j]))
    print("Coste de la solución: " + str(Coste_Final))


    # Graficar el mapa y los puntos
    lista_base_indices = []
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_inter, latitudes_inter, color='green', label='Intermediarios')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    bases = puntos[:numero_bases]
    longitudes_bases, latitudes_bases = zip(*bases)
    for v in range(len(ind_intermediarios)):
            if isinstance(Sol_Final[1], float):
                Sol_Final[1] = list(np.zeros(numero_bases))
            base_indices = [i for i, x in enumerate(Sol_Final[1]) if x == ind_intermediarios[v]]
            for j in base_indices:
                plt.plot([longitudes_bases[j], longitudes_inter[v]],[latitudes_bases[j], latitudes_inter[v]], color='yellow')
            lista_base_indices.extend(base_indices)
    for k in range(Tam_Individuos):
        Sol_Final[0][k] = int(Sol_Final[0][k])
        if k not in lista_base_indices:
            plt.plot([longitudes_bases[k],longitudes_supply_depots[Sol_Final[0][k]]], [latitudes_bases[k], latitudes_supply_depots[Sol_Final[0][k]]],color='red')
        if k in ind_intermediarios:
            plt.plot([longitudes_bases[k],longitudes_supply_depots[Sol_Final[0][k]]], [latitudes_bases[k], latitudes_supply_depots[Sol_Final[0][k]]],color='red')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()
