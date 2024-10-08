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
        Pob_Ini = np.random.randint(0,Num_Max, size=(Fil,Col))  #Son los índices de los SD asignados a cada base
        for i in range(Fil):    #Comprobamos todos los individuos y los reparamos si estuvieran mal
            if(self.Comprobacion_Individuo(Pob_Ini[i], Capacidades)):
                Pob_Ini[i] = self.Reparacion_Mayor_Menor(Pob_Ini[i], Capacidades)
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
            #Indice_Padres = random.sample([j for j in Indices_Validos if j not in Indice_Seleccionado], 2)            # Se elige aleatoriamente el indice de los padres
            #Indice_Seleccionado.extend(Indice_Padres)   #Guardamos los índices elegidos para que no los vuelva a repetir en la siguiente iteración
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

    def Mutacion (self, individuo, Num_Max=None):                                
        aux1 = np.random.randint(0, individuo.shape[0])                         # Se genera número aleatorio para ver la posición que muta
        aux2 = np.random.randint(0,Num_Max)                                   # Se genera el número a modificar
        individuo[aux1] = aux2        
        return individuo
    def Comprobacion_Individuo (self, individuo, capacidades):
        suma_comprobar = list(np.zeros(self.Num_Max))
        for i in range(self.Num_Max):
            indices_bases = [j for j, value in enumerate(individuo) if value == i]  #Obtenemos los indices de las bases asociadas a un SD "i"
            for j in range(len(indices_bases)):
                if np.array_equal(lista_clases_base[indices_bases[j]],lista_clases_SD[i]):   #Comprobamos si tienen los mismos recursos la base asignada y el SD
                    comprobar_capacidades = capacidades[indices_bases[j]]
                    suma_comprobar[i] += comprobar_capacidades
                else:   #Si no, hay que reparar
                    return True
        Caps_Comprobar = [ind_cap for ind_cap, j in enumerate(suma_comprobar) if j > 200]
        if len(Caps_Comprobar) > 0:
            return True
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
            for j in range(len(indices_bases_reparar)):
                if np.array_equal(lista_clases_base[indices_bases_reparar[j]], lista_clases_SD[i]):   #Comprobamos si tienen los mismos recursos la base asignada y el SD
                    capacidades_sd_i = capacidades[indices_bases_reparar[j]]
                    capacidades_sd[i] = capacidades_sd_i
                    suma_capacidades[i] += capacidades_sd[i]    #Almacenamos todas las sumas de las capacidades en un array
                else:   #Si no, buscamos una base que sí los tenga
                    #indices_bases_misma_clase = [k for k,v in enumerate(lista_clases_base) if np.array_equal(v,lista_clases_SD[i])]
                    #Sacamos todas las bases que tengan los mismos recursos que el SD
                    #indice_bases_misma_clase = random.choice(indices_bases_misma_clase) #Elegimos aleatoriamente una de las posibles bases
                    #capacidades_sd_i = capacidades[indice_bases_misma_clase]
                    #capacidades_sd[i] = capacidades_sd_i
                    #suma_capacidades[i] += capacidades_sd[i]  # Almacenamos todas las sumas de las capacidades en un array
                    #individuo[indice_bases_misma_clase] = i  # Cambiamos la asignación de esa base al SD al que debería estar asociada
                    individuo[indices_bases_reparar[j]] = random.choice([k for k, v in enumerate(lista_clases_SD) if np.array_equal(v,lista_clases_base[indices_bases_reparar[j]])])
                    # Asociamos a la base que queremos reparar un SD que coincida con sus recursos
        Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if j > 200]  # Comprobamos en qué SD's se han superado la capacidad

        if len(Caps_SD_Superadas) > 0:
            while True:
                k_2 = np.argsort(suma_capacidades)[::-1]
                k = k_2[0]  # Solucionamos aquella capacidad que sea mas grande
                while True:
                    SD_mismos_recursos = [i for i, v in enumerate(lista_clases_SD) if np.array_equal(v,lista_clases_SD[k]) and i != k] #Sacamos índices de SD con mismos recursos
                    k_3 = random.choice(SD_mismos_recursos)  # Jugamos con uno de los SD con mismos recursos que las bases
                    indices_bases_SD = [j for j, value in enumerate(individuo) if value == k]    #Obtenemos índices de las bases cuya suma de caps supera el umbral
                    indices_resto_bases = [j for j, value in enumerate(individuo) if value == k_3]  # Obtenemos índices de bases para otro SD con mismos recursos
                    capacidades_bases_SD_ordenados = list(np.argsort([capacidades[i] for i in indices_bases_SD])[::-1])
                    indices_bases_SD_ordenados = [indices_bases_SD[i] for i in capacidades_bases_SD_ordenados]

                    if (suma_capacidades[k] > 200 and suma_capacidades[k] < 220) or (suma_capacidades[k] < 200 and suma_capacidades[k] > 180):
                        indice_base_1 = indices_bases_SD_ordenados[len(indices_bases_SD_ordenados)-np.random.randint(1,len(indices_bases_SD_ordenados))] #Cuando se estabilice la suma de capacidades cogemos caps pequeñas
                        lista_filtrada = [value for value in indices_resto_bases if capacidades[value] <= capacidades[indice_base_1]]
                        if lista_filtrada:
                            indice_base_aleatoria_2 = random.choice(lista_filtrada)  # Elección aleatoria de la base del resto de bases
                        else:
                            SD_mismos_recursos_2 = [v for i, v in enumerate(SD_mismos_recursos) if v != k and v != k_3]
                            if not SD_mismos_recursos_2:    #Si no hay otro SD -> Liberamos bases del SD que da problemas
                                e = random.randint(0, 5)
                                f = indices_bases_SD_ordenados[0:e]
                                individuo[f] = k_3  # ... Descargamos algunas bases del SD que nos da problemas sobre el otro (k_3)
                                continue
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
                                        continue
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
                                if not SD_mismos_recursos_2:  # Si no hay otro SD -> Liberamos bases del SD que da problemas
                                    e = random.randint(0, 5)
                                    f = indices_bases_SD_ordenados[0:e]
                                    individuo[f] = k_3  # ... Descargamos algunas bases del SD que nos da problemas sobre el otro (k_3)
                                    continue
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
    numero_supply_depots = 10
    capacidad_maxima = 13
    numero_clases = 3
    Ruta_Puntos = os.path.join(
        r'.\Resultados\Orografia',
        f"Bases_SD_1.csv")
    Ruta_Capacidades = os.path.join(
        r'.\Resultados\Tipos_Recursos_A',
        f"Cap_Bases_SD_1.csv")
    Ruta_Clases_Bases = os.path.join(
        r'.\Resultados\Tipos_Recursos_A',
        f"Clases_Bases_1.csv")
    Ruta_Clases_SD = os.path.join(
        r'.\Resultados\Tipos_Recursos_A',
        f"Clases_SD_1.csv")
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

    if not os.path.exists(Ruta_Clases_Bases):
        lista_clases_base = []
        for i in range(numero_bases):
            vector_clase = np.zeros(numero_clases, dtype=int)
            indice_clase = random.choice([i for i, v in enumerate(vector_clase)])
            vector_clase[indice_clase] = 1
            lista_clases_base.append(vector_clase)
        lista_clases_base = np.array(lista_clases_base)
        np.savetxt(Ruta_Clases_Bases, lista_clases_base, delimiter=',')
    else:
        lista_clases_base = []
        with open(Ruta_Clases_Bases, mode='r') as file:
            csv_reader = csv.reader(file)
            for fila in csv_reader:
                # Convertir cada elemento de la fila a un número (float o int según sea necesario)
                numbers = [float(x) for x in fila]
                lista_clases_base.append(numbers)
            lista_clases_base = np.array(lista_clases_base)

    if not os.path.exists(Ruta_Clases_SD):
        lista_clases_SD = []
        indices_SD_validos = list(np.arange(numero_supply_depots))  # Sacamos lista de índices de cada SD
        indice_seleccionado = []
        for i in range(numero_supply_depots):
            indice_SD = random.sample([j for j in indices_SD_validos if j not in indice_seleccionado], 1)
            indice_seleccionado.extend(indice_SD)
            vector_clase = np.zeros(numero_clases, dtype=int)
            indice_clase = indice_SD[0] % numero_clases  # Lo que hacemos es asegurarnos que todos los SD tienen una clase distinta
            vector_clase[indice_clase] = 1
            lista_clases_SD.append(vector_clase)
        np.savetxt(Ruta_Clases_SD, lista_clases_SD, delimiter=',')
    else:
        lista_clases_SD = []
        with open(Ruta_Clases_SD, mode='r') as file:
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
    Pob_Inicial = Ev1.PoblacionInicial(capacidad_bases, 100, numero_bases, numero_supply_depots,)  #Poblacion inicial -> 100 posibles soluciones -> PADRES
    for i in range(Num_Generaciones):
        print(("Generación: " + str(i + 1)))
        Fitness = Funcion_Fitness(distancias_euclideas, Pob_Inicial)
        Pob_Actual, Costes = Ev1.Seleccion(Pob_Inicial, Fitness)
        Pob_Inicial = Ev1.Cruce(Pob_Actual, capacidad_bases, numero_supply_depots)  # Aplicamos cruce en las soluciones
        print("Coste: " + str(Costes[0]))
        Costes_Generacion.append(Costes[0])
    Sol_Final = Pob_Inicial[0]   #La primera población será la que tenga menor coste
    Coste_Final = Costes[0]
    print("Solución final:")
    for j in range(Tam_Individuos):
        print("Base " + str(j) + "-> SD: " + str(Sol_Final[j]))
    print("Coste de la solución: " + str(Coste_Final))

    # Graficar el mapa y los puntos
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p',s=60,label='Puntos de Suministro')
    fig.show()
    #Evolución del coste de una de las rutas
    coste = plt.figure(figsize=(10, 6))
    plt.plot(Costes_Generacion)
    plt.xlabel('Número de ejecuciones (Genético)')
    plt.ylabel('Distancia (px/m)')
    coste.show()
    # Graficar solución
    colores = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'yellow', 'magenta', 'cyan', 'violet','lime', 'gold', 'silver', 'indigo']
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p',s=60, label='Puntos de Suministro')
    for k in range(numero_supply_depots):
        SD = np.array([i for i, v in enumerate(Sol_Final) if v == k])
        for j in range(len(SD)):
            color = colores[[i for i, valor in enumerate(lista_clases_base[SD[0]]) if valor == 1][0]]
            plt.scatter(longitudes_bases[SD[j]], latitudes_bases[SD[j]], color=color, label='Bases')
        if len(SD) > 0:
            plt.plot([longitudes_bases[SD[0]], longitudes_supply_depots[k]],[latitudes_bases[SD[0]], latitudes_supply_depots[k]], color='red')
        else:
            continue
    plt.xlabel('Distancia Horizontal (px/m)')
    plt.ylabel('Distancia Vertical (px/m)')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.gca().invert_yaxis()
    plt.show()
