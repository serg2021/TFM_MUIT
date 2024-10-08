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
import os
import csv

class EvolutiveClass:
    def __init__(self, Num_Individuos=100, Num_Generaciones=10, Tam_Individuos=200, Num_Max = 10, Prob_Padres=0.5, Prob_Mutacion=0.02, Prob_Cruce=0.5):
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
        poblacion_actual = poblacion_inicial[index,:]   #La población tendrá más soluciones que la inicial debido al cruce
        poblacion_actual = poblacion_actual[0:self.Num_Padres,:]    #Nos quedamos con los mejores individuos
        return poblacion_actual, coste_ordenado

    def Cruce (self, poblacion, capacidades, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        #Indice_Seleccionado = []
        Indices_Validos = list(np.arange(self.Num_Padres))

        for i in range(self.Num_Individuos - self.Num_Padres): #Bucle para generar HIJOS
            Indice_Padres = random.sample(Indices_Validos, 2)
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

    def Cruce_Viajante (self, poblacion, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        #Indice_Seleccionado = []
        Indices_Validos = list(np.arange(self.Num_Padres))

        for i in range(self.Num_Individuos - self.Num_Padres): #Bucle para generar HIJOS
            Indice_Padres = random.sample(Indices_Validos, 2)
            ind_orden_p2 = []
            Padre1 = poblacion[Indice_Padres[0],:]                              # Se coge el padre 1
            Padre2 = poblacion[Indice_Padres[1],:]                              # Se coge el padre 2
            rand_tam_p1 = np.random.randint(1, Num_Max)  # Tamaño de la ventana que se va a copiar en Hijo
            rand_ind_p1 = np.random.randint(0, Num_Max)  # Inicio de la ventana
            while rand_ind_p1 + rand_tam_p1 > Num_Max:  # Nos aseguramos de que la ventana no se salga del individuo -> Puede llegar aún así al final de la solución
                rand_ind_p1 = np.random.randint(0, Num_Max)  # Inicio de la ventana
            Hijo = np.zeros(Num_Max, dtype=int)
            Hijo[rand_ind_p1:rand_ind_p1 + rand_tam_p1] = Padre1[rand_ind_p1:rand_ind_p1 + rand_tam_p1]  # Colocamos la ventana en el Hijo
            for j in range(len(Padre2)):
                if Padre2[j] not in Hijo[rand_ind_p1:rand_ind_p1 + rand_tam_p1]:  # Si el valor de Padre2 no se encuentra en la ventana -> Se añade a la lista
                    ind_orden_p2.append(Padre2[j])
            ind_orden_p2 = np.array(ind_orden_p2, dtype=int)  # Lo pasamos a array
            if rand_ind_p1 + rand_tam_p1 == Num_Max:  # Caso extremo -> La ventana llega justo al final de la solución
                Hijo[:rand_ind_p1] = ind_orden_p2
            elif rand_ind_p1 == 0:  # Caso extremo -> La ventana comienza desde el principio de la solución
                Hijo[rand_ind_p1 + rand_tam_p1:] = ind_orden_p2  # Desde la izquierda de la ventana ponemos en orden los valores que ha encontrado en Padre2
            else:
                Hijo[rand_ind_p1 + rand_tam_p1:] = ind_orden_p2[:len(Hijo[rand_ind_p1 + rand_tam_p1:])]  # Desde la izquierda de la ventana ponemos en orden los valores que ha encontrado en Padre2
                Hijo[:rand_ind_p1] = ind_orden_p2[len(Hijo[rand_ind_p1 + rand_tam_p1:]):]
            #Hijo = np.copy(Padre1)                                              # El hijo va a ser una copia del padre 1
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

    def Reparacion_Mayor_Menor (self, individuo, capacidades): #Sustituimos una base de un SD por otra (aleatoriamente) -> Hasta cumplir restricción
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
    pob = pob.astype(int)
    SD = pob[indices[0]]
    for i in range(len(poblacion)):    #Aplicamos la función fitness a cada solución
        fitness = 0
        indices_orden = list(np.argsort(poblacion[i]))[::-1]  #Sacamos el orden de los índices para verlos de forma consecutiva
        for j in range(len(indices_orden)-1):
            k = j +1
            fitness += distancias[indices_orden[j]][indices_orden[k]]    #Calculo fitness buscando en la matriz de distancias la distancia asociada
        fitness += dist[SD][indices[indices_orden[0]]]
        fitness += dist[SD][indices[indices_orden[len(indices_orden)-1]]]
        fitness = fitness/(len(poblacion[0])+1) #Dividimos entre el número de distancias para normalizar
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
    random.seed(2036)
    np.random.seed(2036)
    Num_Individuos = 100
    Num_Generaciones = 500
    Tam_Individuos = 200
    Prob_Padres = 0.5
    Prob_Mutacion = 0.01
    Prob_Cruce = 0.5

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
    Ruta_Solucion = os.path.join(
        r'.\Resultados\Viajante\Escenarios_Cadena_Viajante',
        f"Solucion_1.csv")
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
            capacidad_bases = np.array(capacidad_bases)
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    latitudes_supply_depots, longitudes_supply_depots = zip(*supply_depots)
    capacidad_supply_depots = np.full(numero_supply_depots,200)

    distancias_euclideas = Distancia_Base_Supply_Depot_2D(bases, supply_depots) #Obtenemos distancias de bases a supply depots
    ### A CONTINUACIÓN, APLICAMOS EL ALGORITMO DESPUÉS DE OBTENER LOS COSTES Y DISTANCIAS

    Sol_Final = []
    if os.path.exists(Ruta_Solucion):   #Cargamos la solución
        with open(Ruta_Solucion, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                Sol_Final.append(numbers[0])
    Sol_Final = np.array(Sol_Final)

    ### AQUÍ COMIENZA EL PROBLEMA DEL VIAJANTE
    Individuos = 200
    Generaciones = 300
    Lista_Sol_Final = []
    Costes_Viajante = 0.0
    Costes_Generacion = []
    contador_aux = 0.0
    for i in range(numero_supply_depots):
        print("SD: " + str(i))
        indices_bases_SD = [j for j, value in enumerate(Sol_Final) if value == i]   #Sacamos índices de las bases asociadas a un SD
        Tam_Indiv = len(indices_bases_SD)
        Num_Orden = len(indices_bases_SD)
        Ev2 = EvolutiveClass(Individuos, Generaciones, len(indices_bases_SD), len(indices_bases_SD), Prob_Padres, Prob_Mutacion,Prob_Cruce) #Objeto de Evolutivo
        Pob_Init = Ev2.PoblacionInicial_Viajante(Individuos, Tam_Indiv, Num_Orden)
        dist_bases_list_SD = []
        bases_SD = [bases[v] for v in indices_bases_SD] #Bases asociadas a un SD
        for x in range(len(indices_bases_SD)):  #Sacamos distancias entre las bases del mismo SD
            distancia_euclidea_SD = Distancia_Base_Supply_Depot_2D(bases_SD,bases[indices_bases_SD[x]])  # Obtenemos distancias de bases con otra base
            dist_bases_list_SD.append(distancia_euclidea_SD)    #Añadimos esas distancias a la lista principal -> Al final obtenemos una diagonal de 0's
        for j in range(Generaciones):
            if j % 50 == 0 and j != 0: #Cada 50 generaciones, reinicializamos la población parcialmente para evitar mínimos locales
                contador = 0
                idx = np.random.randint(1,len(indices_bases_SD), size=(int(numero_bases*0.25)-1,2))
                #Swapping
                for row in idx:
                    while row[0] == row[1]:
                        row[1] = np.random.randint(0, len(indices_bases_SD))
                for row_idx, (idx1,idx2) in enumerate(idx):
                    Pob_Init[row_idx+1, [idx1, idx2]] = Pob_Init[row_idx+1, [idx2, idx1]]
                contador += int(numero_bases*0.25)
                #Block-Swapping
                rand_tam = np.random.randint(1, len(indices_bases_SD) / 4, size=(int(numero_bases * 0.25)))    #Tamaño aleatorio de 2 bloques/solución
                idx_2 = np.random.randint(0, len(indices_bases_SD)/2, size=(int(numero_bases * 0.25)))   #Índice de comienzo del primer bloque
                for x in range(len(idx_2)):
                    while True:
                        idx_2_2 = np.random.randint(0,len(indices_bases_SD) - rand_tam[x]) #Elegimos un índice de bloque que no solape con el primero
                        if (idx_2[x] + rand_tam[x] <= idx_2_2) or (idx_2_2 + rand_tam[x] <= idx_2[x]):
                            break
                    copia_bloque = Pob_Init[x + contador][idx_2_2:idx_2_2+rand_tam[x]].copy()
                    Pob_Init[x + contador][idx_2_2:idx_2_2+rand_tam[x]] = Pob_Init[x + contador][idx_2[x]:idx_2[x]+rand_tam[x]].copy()
                    Pob_Init[x + contador][idx_2[x]:idx_2[x] + rand_tam[x]] = copia_bloque
                contador += int(numero_bases*0.25)
                #Rotación Circular
                rand_circle = np.random.randint(1, len(indices_bases_SD), size=(int(numero_bases * 0.25)))    #Número de rotaciones por solución
                for y in range(len(rand_circle)):
                    Pob_Init[y + contador] = np.hstack((Pob_Init[y + contador][rand_circle[y]:],Pob_Init[y + contador][:rand_circle[y]]))
                contador += int(numero_bases * 0.25)
                #Reinicializamos población restante
                Pob_Init[contador:] = Ev2.PoblacionInicial_Viajante(int(numero_bases*0.25), Tam_Indiv, Num_Orden)
                #Pob_Init[1:int(numero_bases*0.5)] = np.array([dos_opt(elemento) for ind, elemento in enumerate(Pob_Init[1:int(numero_bases*0.5)])]) #2-Opt
            Fitness_Viajante = Funcion_Fitness_Viajante(dist_bases_list_SD, distancias_euclideas, Pob_Init, Sol_Final,indices_bases_SD)
            Pob_Act, Costes_Viajante = Ev2.Seleccion(Pob_Init, Fitness_Viajante)
            Pob_Init = Ev2.Cruce_Viajante(Pob_Act, Num_Orden)
            Costes_Generacion.append(Costes_Viajante[0])
        print("Coste Solución SD " + str(i) + ": " + str(Costes_Viajante[0]))
        contador_aux += Costes_Viajante[0]
        Lista_Sol_Final.append(Pob_Init[0])
    print("Media Costes SD: " + str(contador_aux/numero_supply_depots))

    # Graficar el mapa y los puntos
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', s=60, label='Puntos de Suministro')
    plt.gca().invert_yaxis()
    fig.show()
    #Evolución del coste de una de las rutas
    coste = plt.figure(figsize=(10, 6))
    plt.plot(Costes_Generacion[2700:2999])
    plt.xlabel('Número de ejecuciones (Genético)')
    plt.ylabel('Distancia (px/m)')
    coste.show()
    #Graficamos las rutas óptimas
    colores = ['green', 'magenta', 'red', 'orange', 'purple', 'brown', 'pink', 'yellow', 'black', 'cyan']
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', s=60, label='Puntos de Suministro')
    plt.xlabel('Distancia Horizontal (px/m)')
    plt.ylabel('Distancia Vertical (px/m)')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    for v in range(len(Lista_Sol_Final)-7):
        color = colores[v % len(colores)]   #Un color por cada iteración
        indices_bases_SD = [j for j, value in enumerate(Sol_Final) if value == v]  # Sacamos índices de las bases asociadas a un SD
        indices_ordenados = list(np.argsort(Lista_Sol_Final[v]))    #Ordenamos índices para unir por rectas 2 puntos consecutivos
        indices_bases_SD_ordenados = [indices_bases_SD[i] for i in indices_ordenados]
        plt.plot([longitudes_bases[indices_bases_SD_ordenados[0]], longitudes_supply_depots[v]],
                 [latitudes_bases[indices_bases_SD_ordenados[0]], latitudes_supply_depots[v]], color=color)
        for k in range(0,len(indices_bases_SD_ordenados)-1): #Bucle que recorre los valores
            plt.plot([longitudes_bases[indices_bases_SD_ordenados[k]], longitudes_bases[indices_bases_SD_ordenados[k+1]]],
                     [latitudes_bases[indices_bases_SD_ordenados[k]], latitudes_bases[indices_bases_SD_ordenados[k+1]]], color=color)
    plt.gca().invert_yaxis()
    plt.show()
