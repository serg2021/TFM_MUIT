# -*- coding: utf-8 -*-
"""
@author: Sergio

1º -> Activamos el entorno de conda -> conda activate TFM_MUIT
2º -> Podemos ejecutar el código sin problemas
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
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
        return np.array(Pob_Ini_List)

    def PoblacionInicial_Viajante(self, Fil=None, Col=None, Num_Max=None):
        if Fil == None:
            Fil = self.Num_Individuos
        if Col == None:
            Col = self.Tam_Individuos
        if Num_Max == None:
            Num_Max = self.Num_Max
        Pob_Ini = np.random.randint(0,Num_Max, size=(Fil,Col))  #Son los índices de los SD asignados a cada base
        for i in range(Fil):    #Comprobamos todos los individuos y los reparamos si estuvieran mal
            if(self.Comprobacion_Individuo_Viajante(Pob_Ini[i])):
                Pob_Ini[i] = self.Reparacion_Viajante(Pob_Ini[i])
        return Pob_Ini

    def Seleccion(self, poblacion_inicial, coste):
        index = np.argsort(coste)
        coste_ordenado = np.sort(coste)
        poblacion_actual = poblacion_inicial[index][:]   #La población tendrá más soluciones que la inicial debido al cruce
        poblacion_actual = poblacion_actual[0:self.Num_Padres][:]    #Nos quedamos con los mejores individuos
        return poblacion_actual, coste_ordenado

    def Cruce (self, poblacion, capacidades, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        capacidades_sd = list(np.zeros(self.Num_Max))   #Capacidades de los SD
        #Indice_Seleccionado = []
        Indices_Validos = list(np.arange(self.Num_Padres))

        for v in range(self.Num_Individuos - self.Num_Padres): #Bucle para generar HIJOS
            Indice_Padres = random.sample(Indices_Validos, 2)
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
            poblacion = np.insert(poblacion,self.Num_Padres +v,Hijo, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado
        return poblacion

    def Cruce_Viajante (self, poblacion, Num_Max = None):   #Aplicamo Order Crossover (OX)
        if Num_Max == None:
            Num_Max = self.Num_Max
        Indices_Validos = list(np.arange(self.Num_Padres))

        for i in range(self.Num_Individuos - self.Num_Padres): #Bucle para generar HIJOS
            Indice_Padres = random.sample(Indices_Validos, 2)
            ind_orden_p2 = []
            Padre1 = poblacion[Indice_Padres[0],:]                              # Se coge el padre 1
            Padre2 = poblacion[Indice_Padres[1],:]                              # Se coge el padre 2
            if Num_Max > 4: #Así evitamos errores en el random para que el tamaño de la ventana no sea igual que la solución entera -> No es el objetivo
                rand_tam_p1 = np.random.randint(1, int(Num_Max/2)) #Tamaño de la ventana que se va a copiar en Hijo
            else:
                rand_tam_p1 = np.random.randint(1, Num_Max) #Tamaño de la ventana que se va a copiar en Hijo
            rand_ind_p1 = np.random.randint(0, Num_Max) #Inicio de la ventana
            while rand_ind_p1 + rand_tam_p1 > Num_Max:  #Nos aseguramos de que la ventana no se salga del individuo -> Puede llegar aún así al final de la solución
                rand_ind_p1 = np.random.randint(0, Num_Max)  # Inicio de la ventana
            Hijo = np.zeros(Num_Max, dtype=int)
            Hijo[rand_ind_p1:rand_ind_p1 + rand_tam_p1] = Padre1[rand_ind_p1:rand_ind_p1 + rand_tam_p1] #Colocamos la ventana en el Hijo
            for j in range(len(Padre2)):
                if Padre2[j] not in Hijo[rand_ind_p1:rand_ind_p1 + rand_tam_p1]:    #Si el valor de Padre2 no se encuentra en la ventana -> Se añade a la lista
                    ind_orden_p2.append(Padre2[j])
            ind_orden_p2 = np.array(ind_orden_p2, dtype=int)   #Lo pasamos a array
            if rand_ind_p1 + rand_tam_p1 == Num_Max:    #Caso extremo -> La ventana llega justo al final de la solución
                Hijo[:rand_ind_p1] = ind_orden_p2
            elif rand_ind_p1 == 0:  #Caso extremo -> La ventana conmienza desde el principio de la solución
                Hijo[rand_ind_p1 + rand_tam_p1:] = ind_orden_p2 #Desde la izquierda de la ventana ponemos en orden los valores que ha encontrado en Padre2
            else:
                Hijo[rand_ind_p1 + rand_tam_p1:] = ind_orden_p2[:len(Hijo[rand_ind_p1 + rand_tam_p1:])] #Desde la izquierda de la ventana ponemos en orden los valores que ha encontrado en Padre2
                Hijo[:rand_ind_p1] = ind_orden_p2[len(Hijo[rand_ind_p1 + rand_tam_p1:]):]
            #vector = 1*(np.random.rand(self.Tam_Individuos) > self.Prob_Cruce)  # Se genera un vector para seleccionar los genes del padre 2
            #Hijo[np.where(vector==1)[0]] = Padre2[np.where(vector==1)[0]]       # Los genes seleccionados del padre 2 pasan al hijo
            if(self.Comprobacion_Individuo_Viajante(Hijo)):                    # Se comprueba si hay que reparar el hijo
                 Hijo = self.Reparacion_Viajante(Hijo)
            if np.random.rand() < self.Prob_Mutacion:                           # Se comprueba si el hijo va a mutar
                Hijo = self.Mutacion_Viajante(Hijo, Num_Max)
                Hijo = dos_opt(Hijo)    #Aplicamos Local Search
            poblacion = np.insert(poblacion,self.Num_Padres+i,Hijo, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado
        return poblacion

    def Mutacion_Viajante (self, individuo, Num_Max=None):
        rand_circle = np.random.randint(1, Num_Max)  # Número de rotaciones por solución
        individuo = np.hstack((individuo[rand_circle:], individuo[:rand_circle]))
        #aux = random.sample(list(np.arange(Num_Max)),2)                        # Se generan 2 números aleatorios para ver las posiciones que mutan
        #individuo[aux[0]], individuo[aux[1]] = individuo[aux[1]], individuo[aux[0]]     #Intercambiamos posiciones
        return individuo
    def Mutacion (self, individuo, Num_Max=None):                                
        aux1 = np.random.randint(0, individuo[0].shape[0])                         # Se genera número aleatorio para ver la posición que muta
        aux2 = np.random.randint(0,Num_Max)                                   # Se genera el número a modificar
        individuo[0][aux1] = aux2
        return individuo
    def Comprobacion_Individuo_Viajante (self, individuo):
        set_individuo = set(individuo)  #Lo pasamos a set, ya que en un set no hya elementos duplicados
        if len(individuo) != len(set_individuo):    #Comprobamos longitudes -> Si son distintas, quiere decir que había duplicados
            return True
    def Reparacion_Viajante(self, individuo):   #Lo que haremos será asegurarnos de que no se repiten números en toda la lista
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
    indices_orden = list(np.argsort(individuo))  # Sacamos el orden de los índices para verlos de forma consecutiva
    for j in range(len(indices_orden) - 1):
        k = j + 1
        fitness += distancias[indices_orden[j]][indices_orden[k]]  # Calculo fitness buscando en la matriz de distancias la distancia asociada
    #fitness += dist[SD][indices[indices_orden[0]]]
    #fitness += dist[SD][indices[indices_orden[len(indices_orden) - 1]]]  # Sumamos distancia del camino de vuelta
    fitness = fitness / len(individuo)
    return fitness

def Funcion_Fitness_Viajante(distancias, dist, poblacion, pob, indices):
    lista_fitness = []
    SD = pob[0][indices[0]]
    for i in range(len(poblacion)):    #Aplicamos la función fitness a cada solución
        #contador = 1
        fitness = 0
        indices_orden = list(np.argsort(poblacion[i])) #Sacamos el orden de los índices para verlos de forma consecutiva
        for j in range(len(indices_orden)-1):
            k = j +1
            if indices[indices_orden[j]] in ind_intermediarios:
                #num_bases = np.where(pob[1] == indices[indices_orden[j]])   #Sacamos el número de bases asociadas al intermediario
                #if num_bases != 0:
                #    contador += len(num_bases[0])+1 #Contamos las bases asociadas en la normalización
                fitness += distancias[indices_orden[j]][indices_orden[k]]
                #indice_inter = [v for v, value in enumerate(ind_intermediarios) if value == indices[indices_orden[j]]]  #Sacamos el índice del intermediario para buscarlo en la lista de fitness
                #fitness += Lista_Fitness_Intermediario[indice_inter[0]] #Sumamos fitness de bases asociadas al intermediario en la solución
            else:
                fitness += distancias[indices_orden[j]][indices_orden[k]]    #Calculo fitness buscando en la matriz de distancias la distancia asociada
        fitness += dist[SD][indices[indices_orden[0]]]
        fitness += dist[SD][indices[indices_orden[len(indices_orden)-1]]]   #Sumamos distancia del camino de vuelta
        fitness = fitness/(len(poblacion[0])+1)
        lista_fitness.append(fitness)
    return lista_fitness

def Funcion_Fitness_Viajante_Intermediario(distancias, poblacion):
    lista_fitness = []
    for i in range(len(poblacion)):    #Aplicamos la función fitness a cada solución
        fitness = 0
        indices_orden = list(np.argsort(poblacion[i]))  #Sacamos el orden de los índices para verlos de forma consecutiva
        for j in range(len(indices_orden)-1):
            k = j+1
            fitness += distancias[indices_orden[j]][indices_orden[k]]    #Calculo fitness buscando en la matriz de distancias la distancia asociada
        fitness += distancias[indices_orden[0]][len(indices_orden)]         #Sumamos distancia desde inter hasta base
        fitness += distancias[indices_orden[len(indices_orden)-1]][len(indices_orden)]  #Sumamos distancia de vuelta al inter
        fitness = fitness/(len(poblacion[0])+1)# -> Aquí normalizaríamos, pero perderíamos información de distancias totales en los intermediarios para la solución general
        lista_fitness.append(fitness)
    return lista_fitness

def dos_opt(individuo):    #Mecanismo para hacer Local Search (Cambiamos 2 nodos no adyacentes e invertimos la ruta entre ellos sólo)
    best = individuo.copy()
    if len(individuo) == 2: #Caso extremo de 2 bases para un intermediario
        return individuo
    improved = True
    while improved:
        improved = False
        for i in range(1, len(individuo) - 1):
            for j in range(i + 1, len(individuo)):
                if j - i == 1:  #Evitamos nodos adyacentes
                    continue
                individuo_aux = individuo.copy()
                individuo_aux[i:j] = individuo[j - 1:i - 1:-1]  #Invertimos el orden entre esas 2 subrutas
                if Funcion_Fitness(dist_bases_list_SD, individuo_aux) < Funcion_Fitness(dist_bases_list_SD, individuo):
                    best = individuo_aux
                    improved = True
        individuo = best.copy()
    return best

if __name__ == "__main__":
    # Definicion de los parámetros del genético
    Num_Individuos = 100
    Num_Generaciones = 10
    Tam_Individuos = 200
    Prob_Padres = 0.5
    Prob_Mutacion = 0.01
    Prob_Cruce = 0.5

    Pob_Actual = []
    Lista_Bases_Actual = []
    Costes = []
    numero_bases = 200
    numero_supply_depots = 10
    capacidad_maxima = 20
    Ruta_Puntos = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Cadena_Suministro',
        f"Bases_SD.csv")
    Ruta_Intermediarios = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Cadena_Suministro',
        f"Intermediarios.csv")
    Ruta_Capacidades = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Resultados\Cadena_Suministro',
        f"Cap_Bases_SD.csv")
    Ruta_Solucion = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Combinacion_Problemas',
        f"Solucion.csv")
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
    longitudes_bases, latitudes_bases = zip(*bases)
    longitudes_inter, latitudes_inter = zip(*intermediarios)
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
    longitudes_supply_depots, latitudes_supply_depots = zip(*supply_depots)
    capacidad_supply_depots = np.full(numero_supply_depots,200)

    distancias_euclideas = Distancia_Base_Supply_Depot_2D(bases_inter, supply_depots) #Obtenemos distancias de bases a supply depots
    ### A CONTINUACIÓN, APLICAMOS EL ALGORITMO DESPUÉS DE OBTENER LOS COSTES Y DISTANCIAS

    for v in range(Tam_Individuos):
        if v in ind_intermediarios:
            print("Intermediario: " + str(v) + " -> Capacidad: " + str(capacidad_bases[v]))

    Sol_Final = []
    if os.path.exists(Ruta_Solucion):   #Cargamos la solución
        with open(Ruta_Solucion, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                Sol_Final.append(numbers)
            Sol_Final = np.array(Sol_Final, dtype=int)
    Sol_Final = list(Sol_Final)

    lista_base_indices = []
    #Graficar solución
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_inter, latitudes_inter, color='green', label='Intermediarios')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    bases = puntos[:numero_bases]
    longitudes_bases, latitudes_bases = zip(*bases)
    for v in range(len(ind_intermediarios)):
        base_indices = [i for i, x in enumerate(Sol_Final[1]) if x == ind_intermediarios[v]]
        for j in base_indices:
            plt.plot([longitudes_bases[j], longitudes_inter[v]],[latitudes_bases[j], latitudes_inter[v]], color='yellow')
        lista_base_indices.extend(base_indices)
    for k in range(numero_supply_depots):
        SD = [i for i, v in enumerate(Sol_Final[0]) if v == k]  # Sacamos bases asociadas a un SD
        if len(SD) > 0:
            for l in range(len(SD)):
                if SD[l] in ind_intermediarios:
                    plt.plot([longitudes_bases[SD[l]],longitudes_supply_depots[Sol_Final[0][SD[l]]]], [latitudes_bases[SD[l]], latitudes_supply_depots[Sol_Final[0][SD[l]]]],color='red')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()

    # AQUÍ COMIENZA EL VIAJANTE DE INTERMEDIARIOS

    #Primero hacemos viajante para todos aquellos intermediarios que tengan una base o más
    Individuos = 100
    Generaciones = 250
    Lista_Rutas_Intermediario = []
    Lista_Fitness_Intermediario = []
    Costes_Viajante = 0.0
    Num_Inter = len(ind_intermediarios) #Número de intermediarios
    for i in range(Num_Inter):  #Bucle para cada intermediario
        print("Intermediario: " + str(i+1))
        indices_bases_inter = [j for j, value in enumerate(Sol_Final[1]) if value == ind_intermediarios[i]]   #Sacamos índices de las bases asociadas a un intermediario
        if len(indices_bases_inter) == 0:   #No hay bases asociadas a ese intermediario
            Lista_Rutas_Intermediario.append(0.0)  # Guardamos un valor nulo
            Lista_Fitness_Intermediario.append(0.0) #Guardamos un valor nulo -> Lo hacemos para no perder el orden de los intermediarios
            continue    #Siguiente iteración
        Tam_Indiv = len(indices_bases_inter)
        Num_Orden = len(indices_bases_inter)
        Ev2 = EvolutiveClass(Individuos, Generaciones, len(indices_bases_inter), len(indices_bases_inter), Prob_Padres, Prob_Mutacion,Prob_Cruce) #Objeto de Evolutivo
        Pob_Init = Ev2.PoblacionInicial_Viajante(Individuos, Tam_Indiv, Num_Orden)
        dist_bases_list_SD = []
        dist_bases_inter = [bases[v] for v in indices_bases_inter] #Bases asociadas a un intermediario
        dist_bases_inter.append(bases[ind_intermediarios[i]])   #Añadimos también el propio intermediario
        for x in range(len(indices_bases_inter)):  #Sacamos distancias entre las bases del mismo SD
            distancia_euclidea_inter = Distancia_Base_Supply_Depot_2D(dist_bases_inter,bases[indices_bases_inter[x]])  # Obtenemos distancias de bases con otra base
            dist_bases_list_SD.append(distancia_euclidea_inter)    #Añadimos esas distancias a la lista principal -> Al final obtenemos una diagonal de 0's
        if len(indices_bases_inter) == 1: #Una base asociada a ese intermediario
            fitness_1 = 0.0
            fitness_1 += 2.0 * dist_bases_list_SD[0][len(dist_bases_list_SD)]
            Array_Indices = np.array(indices_bases_inter)
            Ruta_Orden = np.argsort(Pob_Init[0])  # Ordenamos las bases
            Ruta_Inter = Array_Indices[Ruta_Orden]
            Lista_Rutas_Intermediario.append(Ruta_Inter)  # Guardamos la ruta de ese intermediario
            Lista_Fitness_Intermediario.append(fitness_1)
            continue
        for j in range(Generaciones):
            Fitness_Viajante = Funcion_Fitness_Viajante_Intermediario(dist_bases_list_SD, Pob_Init)
            Pob_Act, Costes_Viajante = Ev2.Seleccion(Pob_Init, Fitness_Viajante)
            Pob_Init = Ev2.Cruce_Viajante(Pob_Act, Num_Orden)
        print("Coste Solución " + str(i) + ": " + str(Costes_Viajante[0]))
        Array_Indices = np.array(indices_bases_inter)
        Ruta_Inter = Array_Indices[Pob_Init[0]]
        Lista_Rutas_Intermediario.append(Ruta_Inter)    #Guardamos la ruta de ese intermediario
        Lista_Fitness_Intermediario.append(Costes_Viajante[0])

    #AQUÍ COMIENZA EL VIAJANTE NORMAL CONTANDO LOS FITNESS DE LOS INTERMEDIARIOS
    #Hay que mirar a ver cómo hacemos el plot de las rutas teniendo en cuenta las de las bases de los intermediarios
    Individuos = 200
    Generaciones = 1000
    Lista_Sol_Final = []
    Costes_Viajante = 0.0
    Costes_Generacion = []
    for k in range(numero_supply_depots):
        print("SD: " + str(k+1))
        indices_bases_SD = np.array([j for j, value in enumerate(Sol_Final[0]) if value == k])  #Sacamos índices de las bases asociadas a un SD
        indices_bases_inter = np.array([j for j, value in enumerate(indices_bases_SD) if Sol_Final[1][value] == numero_bases])  #Sacamos bases e inters directos a un SD
        Tam_Indiv = len(indices_bases_inter)
        Num_Orden = len(indices_bases_inter)
        Ev2 = EvolutiveClass(Individuos, Generaciones, len(indices_bases_inter), len(indices_bases_inter), Prob_Padres, Prob_Mutacion,Prob_Cruce) #Objeto de Evolutivo
        Pob_Init = Ev2.PoblacionInicial_Viajante(Individuos, Tam_Indiv, Num_Orden)
        dist_bases_list_SD = []
        bases_SD = [bases[v] for v in indices_bases_SD[indices_bases_inter]] #Bases asociadas a un SD
        for x in range(len(indices_bases_inter)):  #Sacamos distancias entre las bases del mismo SD
            distancia_euclidea_SD = Distancia_Base_Supply_Depot_2D(bases_SD,bases[indices_bases_SD[indices_bases_inter[x]]])  # Obtenemos distancias de bases con otra base
            dist_bases_list_SD.append(distancia_euclidea_SD)    #Añadimos esas distancias a la lista principal -> Al final obtenemos una diagonal de 0's
        for j in range(Generaciones):
            if j % 250 == 0 and j != 0: #Cada 50 generaciones, reinicializamos la población parcialmente para evitar mínimos locales
                contador = 0
                idx = np.random.randint(1,len(indices_bases_inter), size=(int(numero_bases*0.25)-1,2))
                #Swapping
                for row in idx:
                    while row[0] == row[1]:
                        row[1] = np.random.randint(0, len(indices_bases_inter))
                for row_idx, (idx1,idx2) in enumerate(idx):
                    Pob_Init[row_idx+1, [idx1, idx2]] = Pob_Init[row_idx+1, [idx2, idx1]]
                contador += int(numero_bases*0.25)
                #Block-Swapping
                rand_tam = np.random.randint(1, len(indices_bases_inter) / 4, size=(int(numero_bases * 0.25)))    #Tamaño aleatorio de 2 bloques/solución
                idx_2 = np.random.randint(0, len(indices_bases_inter)/2, size=(int(numero_bases * 0.25)))   #Índice de comienzo del primer bloque
                for x in range(len(idx_2)):
                    while True:
                        idx_2_2 = np.random.randint(0,len(indices_bases_inter) - rand_tam[x]) #Elegimos un índice de bloque que no solape con el primero
                        if (idx_2[x] + rand_tam[x] <= idx_2_2) or (idx_2_2 + rand_tam[x] <= idx_2[x]):
                            break
                    copia_bloque = Pob_Init[x + contador][idx_2_2:idx_2_2+rand_tam[x]].copy()
                    Pob_Init[x + contador][idx_2_2:idx_2_2+rand_tam[x]] = Pob_Init[x + contador][idx_2[x]:idx_2[x]+rand_tam[x]].copy()
                    Pob_Init[x + contador][idx_2[x]:idx_2[x] + rand_tam[x]] = copia_bloque
                contador += int(numero_bases*0.25)
                #Rotación Circular
                rand_circle = np.random.randint(1, len(indices_bases_inter), size=(int(numero_bases * 0.25)))    #Número de rotaciones por solución
                for y in range(len(rand_circle)):
                    Pob_Init[y + contador] = np.hstack((Pob_Init[y + contador][rand_circle[y]:],Pob_Init[y + contador][:rand_circle[y]]))
                contador += int(numero_bases * 0.25)
                #Reinicializamos población restante
                Pob_Init[contador:] = Ev2.PoblacionInicial_Viajante(int(numero_bases*0.25), Tam_Indiv, Num_Orden)
                Pob_Init[1:int(numero_bases*0.5)] = np.array([dos_opt(elemento) for ind, elemento in enumerate(Pob_Init[1:int(numero_bases*0.5)])]) #2-Opt
            Fitness_Viajante = Funcion_Fitness_Viajante(dist_bases_list_SD, distancias_euclideas, Pob_Init, Sol_Final,indices_bases_SD[indices_bases_inter])
            Pob_Act, Costes_Viajante = Ev2.Seleccion(Pob_Init, Fitness_Viajante)
            np.random.shuffle(Pob_Act[1:])     #Salteamos la población para no concentrar los mismos individuos en una zona -> Todos menos el mejor (elitismo)
            Pob_Init = Ev2.Cruce_Viajante(Pob_Act, Num_Orden)
            Costes_Generacion.append(Costes_Viajante[0])
        print("Coste Solución SD " + str(k+1) + ": " + str(Costes_Viajante[0]))
        Lista_Sol_Final.append(indices_bases_SD[indices_bases_inter][Pob_Init[0]])  #Guardamos la ruta de la solución

    # Graficar el mapa y los puntos
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_inter, latitudes_inter, color='green', label='Intermediarios')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p',label='Puntos de Suministro')
    fig.show()
    # Evolución del coste de una de las rutas
    coste = plt.figure(figsize=(10, 6))
    plt.plot(Costes_Generacion)
    coste.show()
    #Graficamos las rutas óptimas
    colores = ['green', 'magenta', 'red', 'orange', 'purple', 'brown', 'pink', 'yellow', 'black', 'cyan']
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_inter, latitudes_inter, color='green', label='Intermediarios')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p',label='Puntos de Suministro')
    for v in range(len(Lista_Sol_Final)):
        color = colores[v % len(colores)]   #Un color por cada iteración
        plt.plot([longitudes_bases[Lista_Sol_Final[v][0]], longitudes_supply_depots[v]],
                 [latitudes_bases[Lista_Sol_Final[v][0]], latitudes_supply_depots[v]], color=color)
        for k in range(0,len(Lista_Sol_Final[v])-1): #Bucle que recorre los valores
            if Lista_Sol_Final[v][k] in ind_intermediarios and Lista_Sol_Final[v][k+1] in ind_intermediarios: #Si los siguientes son intermediarios
                indice_lista_rutas_1, indice_lista_rutas_2 = np.where(ind_intermediarios == Lista_Sol_Final[v][k])[0], np.where(ind_intermediarios == Lista_Sol_Final[v][k+1])[0] #Buscamos el índice del intermediario
                plt.plot([longitudes_inter[indice_lista_rutas_1[0]],longitudes_inter[indice_lista_rutas_2[0]]],
                         [latitudes_inter[indice_lista_rutas_1[0]],latitudes_inter[indice_lista_rutas_2[0]]], color=color)
                if not isinstance(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]], float): #Tiene bases asociadas
                    if len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) == 1:   #Sólo una base asociada
                        plt.plot([longitudes_inter[indice_lista_rutas_1[0]],longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]],
                                 [latitudes_inter[indice_lista_rutas_1[0]],latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]], color=color)
                    else:   #Varias bases asociadas
                        plt.plot([longitudes_inter[indice_lista_rutas_1[0]],longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]],
                                [latitudes_inter[indice_lista_rutas_1[0]],latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]], color=color)
                        for l in range(len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]])-1):
                            plt.plot([longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l]],longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l+1]]],
                                    [latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l]],latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l+1]]], color=color)
                        plt.plot([longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]])-1]], longitudes_bases[Lista_Sol_Final[v][k]]],
                                 [latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]])-1]], latitudes_bases[Lista_Sol_Final[v][k]]], color=color)
            elif Lista_Sol_Final[v][k] in ind_intermediarios:   #Si el primero es intermediario y el siguiente no lo es
                indice_lista_rutas_1 = np.where(ind_intermediarios == Lista_Sol_Final[v][k])[0]  # Buscamos el índice del intermediario
                if not isinstance(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]],float): # Tiene bases asociadas
                    if len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) == 1:  # Sólo una base asociada
                        plt.plot([longitudes_inter[indice_lista_rutas_1[0]],longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]],
                                 [latitudes_inter[indice_lista_rutas_1[0]],latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]], color=color)
                    else:  # Varias bases asociadas
                        plt.plot([longitudes_inter[indice_lista_rutas_1[0]],longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]],
                                 [latitudes_inter[indice_lista_rutas_1[0]],latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][0]]], color=color)
                        for l in range(len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) - 1):
                            plt.plot([longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l]],longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l + 1]]],
                                     [latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l]],latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][l + 1]]],
                                     color=color)
                        plt.plot([longitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) - 1]],
                                  longitudes_inter[indice_lista_rutas_1[0]]],
                                 [latitudes_bases[Lista_Rutas_Intermediario[indice_lista_rutas_1[0]][len(Lista_Rutas_Intermediario[indice_lista_rutas_1[0]]) - 1]],
                                  latitudes_inter[indice_lista_rutas_1[0]]], color=color)
                plt.plot([longitudes_inter[indice_lista_rutas_1[0]],longitudes_bases[Lista_Sol_Final[v][k+1]]],
                         [latitudes_inter[indice_lista_rutas_1[0]],latitudes_bases[Lista_Sol_Final[v][k+1]]], color=color)
            else:   #Si no es intermediario
                if Lista_Sol_Final[v][k+1] in ind_intermediarios:   #Si el siguiente es intermediario
                    indice_lista_rutas_2 = np.where(ind_intermediarios == Lista_Sol_Final[v][k+1])[0]  # Buscamos el índice del intermediario
                    plt.plot([longitudes_bases[Lista_Sol_Final[v][k]], longitudes_inter[indice_lista_rutas_2[0]]],
                         [latitudes_bases[Lista_Sol_Final[v][k]], latitudes_inter[indice_lista_rutas_2[0]]], color=color)
                elif Lista_Sol_Final[v][k+1] not in ind_intermediarios:  # Si el siguiente es base
                    plt.plot([longitudes_bases[Lista_Sol_Final[v][k]], longitudes_bases[Lista_Sol_Final[v][k+1]]],
                             [latitudes_bases[Lista_Sol_Final[v][k]], latitudes_bases[Lista_Sol_Final[v][k+1]]],color=color)
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()


