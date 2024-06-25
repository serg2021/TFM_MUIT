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
        Pob_Ini = list(np.random.randint(0,Num_Max, size=(Fil,Col)))  #Son los índices de los SD asignados a cada base
        indices_pob_ini = random.sample(list(np.arange(Col)), np.random.randint(0,Col)) #Al azar, elegimos un número de índices que tendrán sublsitas de SD
        for j in indices_pob_ini:
            Pob_Ini[j] = []
            while True:
                if random.choice([True,False]):  # Generamos aleatoriamente
                    Pob_Ini[j].append(np.random.randint(0,Num_Max))
                if len(Pob_Ini[j]) != 0:  # Si me saca una lista vacía, le obligo a intentarlo de nuevo
                    break
        for i in range(Fil):    #Comprobamos todos los individuos y los reparamos si estuvieran mal
            if(self.Comprobacion_Individuo(Pob_Ini[i], Capacidades)):
                Pob_Ini[i] = self.Reparacion_Mayor_Menor(Pob_Ini[i], Capacidades)
        return Pob_Ini

    def Seleccion(self, poblacion_inicial, coste):
        index = np.argsort(coste)
        coste_ordenado = np.sort(coste)
        poblacion_actual = poblacion_inicial[index,:]   #La población tendrá más soluciones que la inicial debido al cruce
        poblacion_actual = poblacion_actual[0:self.Num_Individuos,:]    #Nos quedamos con los mejores individuos
        return poblacion_actual, coste_ordenado

    def Cruce (self, poblacion, capacidades, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        #Indice_Seleccionado = []
        Indices_Validos = list(np.arange(self.Num_Individuos))

        for i in range(self.Num_Individuos - self.Num_Padres): #Bucle para generar HIJOS
            Indice_Padres = random.sample(Indices_Validos, 2)
            #Indice_Padres = random.sample([j for j in Indices_Validos if j not in Indice_Seleccionado], 2)            # Se elige aleatoriamente el indice de los padres
            #Indice_Seleccionado.extend(Indice_Padres)   #Guardamos los índices elegidos para que no los vuelva a repetir en la siguiente iteración
            Padre1 = poblacion[Indice_Padres[0],:]                              # Se coge el padre 1
            Padre2 = poblacion[Indice_Padres[1],:]                              # Se coge el padre 2
            Hijo = list(np.copy(Padre1))                                              # El hijo va a ser una copia del padre 1
            vector = 1*(np.random.rand(self.Tam_Individuos) > self.Prob_Cruce)  # Se genera un vector para seleccionar los genes del padre 2
            Hijo[np.where(vector==1)[0]] = Padre2[np.where(vector==1)[0]]       # Los genes seleccionados del padre 2 pasan al hijo
            if np.random.rand() < self.Prob_Mutacion:                           # Se comprueba si el hijo va a mutar
                Hijo = self.Mutacion(Hijo, Num_Max)
            if(self.Comprobacion_Individuo(Hijo, capacidades)):                    # Se comprueba si hay que reparar el hijo
                 Hijo = self.Reparacion_Mayor_Menor(Hijo, capacidades)
            poblacion = np.insert(poblacion,self.Num_Padres+i,Hijo, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado
        return poblacion

    def Mutacion (self, individuo, Num_Max=None):
        aux1 = np.random.randint(0, len(individuo))                         # Se genera número aleatorio para ver la posición que muta
        if isinstance(individuo[aux1], list):  #Si la base a mutar está asociada a varios SD, cambiamos uno de esos SD
            aux2 = np.random.randint(0, Num_Max)  # Se genera el número a modificar
            aux3 = np.random.randint(0, numero_clases)  # Se genera el SD a modificar
            individuo[aux1][aux3] = aux2
        else:
            aux2 = np.random.randint(0,Num_Max)                                   # Se genera el número a modificar
            individuo[aux1] = aux2
        return individuo
    def Comprobacion_Individuo (self, individuo, capacidades):
        suma_comprobar = list(np.zeros(self.Num_Max))
        suma_comprobar_clase = list(np.zeros(numero_clases, dtype=int))
        for i in range(numero_supply_depots):   #Ponemos la suma de capacidades para cada SD dividida en la suma de capacidades más pequeñas referentes a clases
            suma_comprobar[i] = suma_comprobar_clase
        for i in range(self.Num_Max):   #Bucle para cada SD
            indices_bases = []
            for j, value in enumerate(individuo):
                if isinstance(value, list):
                    # Si el elemento es una sublista, verificamos si tiene el SD
                    if i in value:
                        indices_bases.append(j)
                else:
                    # Si el elemento no es una sublista, verificamos si tiene el SD
                    if value == i:
                        indices_bases.append(j)
            for j in range(len(indices_bases)): #Bucle para base asociada a ese SD
                ind_clases = [k for k,v in enumerate(lista_clases_base[indices_bases[j]]) if v != 0]    #Sacamos qué clases tiene esa base
                for k in ind_clases:    #Bucle para cada clase de esa base
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
        suma_capacidades_clase = list(np.zeros(numero_clases, dtype=int))
        for i in range(numero_supply_depots):  # Ponemos la suma de capacidades para cada SD dividida en la suma de capacidades más pequeñas referentes a clases
            suma_capacidades[i] = suma_capacidades_clase
            capacidades_sd[i] = suma_capacidades_clase
        if any(isinstance(x, list) for x in individuo):
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
                for j in range(len(indices_bases_reparar)):
                    ind_clases = [k for k, v in enumerate(lista_clases_base[indices_bases_reparar[j]]) if v != 0]  # Sacamos qué clases tiene esa base
                    for k in ind_clases:  # Bucle para cada clase de esa base
                        capacidades_sd_i = capacidades[indices_bases_reparar[j]][k]
                        capacidades_sd[SD_ind][k] = capacidades_sd_i
                        suma_capacidades[SD_ind][k] += capacidades_sd[SD_ind][k]    #Almacenamos todas las sumas de las capacidades en un array
                Caps_Clase_Comprobar = [t for t, value in enumerate(suma_capacidades[SD_ind]) if value > 200 / numero_clases]
                if len(Caps_Clase_Comprobar) > 0:  # Si para un SD, se supera el umbral de al menos una clase... -> Reparación
                    while True:
                        k_2 = np.argsort(suma_capacidades[SD_ind])[::-1]
                        k = k_2[0]  # Solucionamos aquella capacidad que sea mas grande de las clases
                        while True: #Bucle infinito hasta que no dé errores
                            SDs = list(np.arange(numero_supply_depots))  #Sacamos lista de SDs
                            k_3 = random.choice(SDs)    #Elegimos uno al azar
                            indices_bases_SD = []   #Bases asociadas al SD de la Cap más grande
                            indices_resto_bases = []    #Bases asociadas al otro SD
                            for j, value in enumerate(individuo):  # Sacamos bases asociadas a un SD
                                if isinstance(value, list):
                                    # Si el elemento es una sublista, verificamos si tiene el SD
                                    if SD_ind in value:
                                        indices_bases_SD.append(j)
                                    if k_3 in value:
                                        indices_resto_bases.append(j)
                                else:
                                    # Si el elemento no es una sublista, verificamos si tiene el SD
                                    if value == SD_ind:
                                        indices_bases_SD.append(j)
                                    if value == k_3:
                                        indices_resto_bases.append(j)
                            capacidades_bases_SD_ordenados = list(np.argsort([capacidades[i][k] for i in indices_bases_SD])[::-1])
                            #Ordenamos índices de capacidades de esa clase de mayor a menor
                            indices_bases_SD_ordenados = [indices_bases_SD[i] for i in capacidades_bases_SD_ordenados]
                            #Ordenamos los índices de las bases al SD de la Cap más grande de mayor a menor

                            indice_base_1 = indices_bases_SD_ordenados[0]   #Cogemos la base con mayor cap de esa clase
                            lista_filtrada = [value for value in indices_resto_bases if capacidades[value] <= capacidades[indice_base_1]]
                            if lista_filtrada:
                                indice_base_aleatoria_2 = random.choice(lista_filtrada)  # Elección aleatoria de la base del resto de bases
                            else:
                                SD_mismos_recursos = [v for i, v in enumerate(SDs) if v != k_3[0] and v != k_3[1]]    #Sacamos lista del resto de SDs
                                lista_ind = []
                                while True:  # Cotejamos que haya bases de la misma clase en otros SD
                                    k_4 = random.choice(SD_mismos_recursos)  # Jugamos con uno de los SD con mismos recursos que las bases
                                    indices_resto_bases = []  # Bases asociadas al otro SD
                                    for j, value in enumerate(individuo):  # Sacamos bases asociadas a un SD
                                        if isinstance(value, list):
                                            # Si el elemento es una sublista, verificamos si tiene el SD
                                            if k_4 in value and k_4 not in lista_ind:
                                                indices_resto_bases.append(j)
                                        else:
                                            # Si el elemento no es una sublista, verificamos si tiene el SD
                                            if value == k_4 and value not in lista_ind:
                                                indices_resto_bases.append(j)
                                    if indices_resto_bases:
                                        break
                                    else:  # Si no hay, la añadimos a una lista
                                        lista_ind.append(k_4)
                                        if len(lista_ind) == len(SDs):  # Cuando el tamaño de lista sea igual que SD_mismos_recursos...
                                            e = random.randint(0, 5)
                                            f = indices_bases_SD_ordenados[0:e]
                                            for i in f:
                                                if isinstance(individuo[i], list):  #Si es una sublista, buscamos índice donde esté el valor de ese SD
                                                    indice = [j for j,v in enumerate(individuo[i]) if v == SD_ind]
                                                    individuo[i][indice] = k_3  #Lo cambiamos por el otro SD
                                                else:
                                                    individuo[i] = k_3  # ... Descargamos algunas bases del SD que nos da problemas sobre el otro (k_3)
                                        else:
                                            continue
                                indice_base_aleatoria_2 = random.choice(indices_resto_bases)
                            if isinstance(individuo[indice_base_1], int) and isinstance(individuo[indice_base_aleatoria_2], int):
                                individuo[indice_base_1], individuo[indice_base_aleatoria_2] = individuo[indice_base_aleatoria_2], individuo[indice_base_1]  # Intercambio posiciones de las bases
                            elif isinstance(individuo[indice_base_1], list) and isinstance(individuo[indice_base_aleatoria_2], list):
                                individuo[indice_base_1][SD_ind], individuo[indice_base_aleatoria_2][k_3] = individuo[indice_base_aleatoria_2][k_3], individuo[indice_base_1][SD_ind]
                            elif isinstance(individuo[indice_base_1], list) or isinstance(individuo[indice_base_aleatoria_2], list):
                                if isinstance(individuo[indice_base_1], list):
                                    individuo[indice_base_1][SD_ind], individuo[indice_base_aleatoria_2] = individuo[indice_base_aleatoria_2],individuo[indice_base_1][SD_ind]  # Intercambio posiciones de las bases
                                elif isinstance(individuo[indice_base_aleatoria_2], list):
                                    individuo[indice_base_1], individuo[indice_base_aleatoria_2][k_3] = individuo[indice_base_aleatoria_2][k_3],individuo[indice_base_1] # Intercambio posiciones de las bases

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
                            for j in range(len(indices_bases_SD)):
                                ind_clases = [k for k, v in enumerate(lista_clases_base[indices_bases_SD[j]]) if v != 0]  # Sacamos qué clases tiene esa base
                                for s in range(numero_clases):  #Inicializamos a 0 todas las sumas de capacidades para la comprobación
                                    suma_capacidades[SD_ind][s] = 0
                                for t in ind_clases:  # Bucle para cada clase de esa base
                                    capacidades_sd_i = capacidades[indices_bases_SD[j]][t]
                                    capacidades_sd[SD_ind][t] = capacidades_sd_i
                                    suma_capacidades[SD_ind][t] += capacidades_sd[SD_ind][t]  # Almacenamos todas las sumas de las capacidades en un array
                            Caps_Clase_SD = [t for t, value in enumerate(suma_capacidades[SD_ind]) if value > 200 / numero_clases]
                            if len(Caps_Clase_SD) == 0:
                                break
                            else:
                                continue
                        Caps_Clase_Comprobar = [t for t, value in enumerate(suma_capacidades[SD_ind]) if value > 200 / numero_clases]
                        if len(Caps_Clase_Comprobar) == 0:
                            break
                    Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if sum(j) > 200]  # Comprobamos en qué SD's se han superado la capacidad
                    if len(Caps_SD_Superadas) > 0:  #Al finalizar con un SD -> Pasamos al siguiente
                        SD_ind += 1
                        if SD_ind == numero_supply_depots:
                            SD_ind = 0
                    else:
                        break
                else:
                    SD_ind += 1
                    if SD_ind == numero_supply_depots:
                        Caps_SD_Superadas = [ind_cap for ind_cap, j in enumerate(suma_capacidades) if sum(j) > 200]  # Comprobamos en qué SD's se han superado la capacidad
                        if len(Caps_SD_Superadas) > 0:
                            SD_ind = 0
                        else:
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
    Num_Individuos = 100
    Num_Generaciones = 500
    Tam_Individuos = 200
    Prob_Padres = 0.1
    Prob_Mutacion = 0.01
    Prob_Cruce = 0.5

    Pob_Actual = []
    Costes = []
    numero_bases = 200
    numero_supply_depots = 30
    capacidad_maxima = 20
    numero_clases = 5
    puntos = list(Puntos_Sin_Repetir(numero_bases+numero_supply_depots))
    supply_depots = puntos[-numero_supply_depots:]
    bases = puntos[:numero_bases]
    longitudes_bases, latitudes_bases = zip(*bases)
    capacidad_bases = list(np.random.randint(1, capacidad_maxima, size=(numero_bases)))
    indices_capacidad_bases = sorted(range(len(capacidad_bases)), key=lambda i: capacidad_bases[i])
    longitudes_supply_depots, latitudes_supply_depots = zip(*supply_depots)
    capacidad_supply_depots = list(np.full(numero_supply_depots,200))

    lista_clases_base = []
    clases = ['A', 'B', 'C', 'D', 'E']
    for i in range(numero_bases):
        while True:
            vector_clase = list(np.zeros(numero_clases))
            caps_bases_clases = list(np.zeros(numero_clases, dtype=int))
            cont = 0
            for j in range(len(clases)):
                if random.choice([True, False]):    #Generamos aleatoriamente bases que pidan un conjunto de recursos distintos entre sí
                    vector_clase[j] = clases[j]
                    cont += 1
            if any(isinstance(x,str) for x in vector_clase):    #Si me saca un array a 0's, le obligo a volver a hacerlo hasta que sea diferente
                break
        if capacidad_bases[i] < cont:
            capacidad_bases[i] = cont   #Hacemos esto para que, si la capacidad de la base es tan baja que no da para todas las clases, forzamos a que tenga más
        if capacidad_bases[i] % cont == 0:  #Quiere decir que la capacidad se reparte de forma equitativa entre las clases
            for k in range(len(clases)):
                if vector_clase[k] == 0:
                    caps_bases_clases[k] = 0
                else:
                    caps_bases_clases[k] = int(capacidad_bases[i]/cont)
        elif capacidad_bases[i] % cont > 0:  #Quiere decir que la capacidad de una clase será 1 unidad mayor
            indices_clases = [j for j, v in enumerate(vector_clase) if v != 0]  # Sacamos índices donde hay clases
            ind = random.choice(indices_clases) #Elegimos uno aleatoriamente
            caps_bases_clases[ind] = int(capacidad_bases[i]/cont) + (capacidad_bases[i] % cont)  #Para esa clase habrá un poco más que para el resto
            for k in range(len(clases)):
                if vector_clase[k] == 0:
                    caps_bases_clases[k] = 0
                elif k != ind:
                    caps_bases_clases[k] = int(capacidad_bases[i]/cont)
        capacidad_bases[i] = tuple(caps_bases_clases)
        lista_clases_base.append(vector_clase)

    lista_clases_SD = []
    for i in range(numero_supply_depots):
        caps_SD_clases = []
        vector_clase = list(np.zeros(numero_clases, dtype=int))
        for j in range(len(clases)):    #Todos los SD podrán dar todo tipo de recursos
            vector_clase[j] = clases[j]
            caps_SD_clases.append(200/len(clases))
        lista_clases_SD.append(vector_clase)
        capacidad_supply_depots[i] = tuple(caps_SD_clases)  #Como un SD tiene un Cap Máxima, cada clase también lo tendrá
                                                            #Suponemos que la capacidad del SD se reparte de forma equitativa para cada tipo de recurso

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
    #Ev1.ImprimirInformacion()
    Pob_Inicial = Ev1.PoblacionInicial(capacidad_bases, 100, numero_bases, numero_supply_depots,)  #Poblacion inicial -> 100 posibles soluciones -> PADRES
    for i in range(Num_Generaciones):
        print(("Generación: " + str(i + 1)))
        Pob_Actual = Ev1.Cruce(Pob_Inicial, capacidad_bases, numero_supply_depots)   #Aplicamos cruce en las soluciones
        Fitness = Funcion_Fitness(distancias_euclideas, Pob_Actual)
        Pob_Inicial, Costes = Ev1.Seleccion(Pob_Actual,Fitness)
    Sol_Final = Pob_Inicial[0]   #La primera población será la que tenga menor coste
    Coste_Final = Costes[0]
    print("Solución final:")
    for j in range(Tam_Individuos):
        print("Base " + str(j) + "-> SD: " + str(Sol_Final[j]))
    print("Coste de la solución: " + str(Coste_Final))
    # Graficar el mapa y los puntos
    colores = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'yellow', 'magenta', 'cyan', 'violet','lime', 'gold', 'silver', 'indigo']
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    for k in range(numero_supply_depots):
        SD = np.array([i for i, v in enumerate(Sol_Final) if v == k])
        color = colores[k]  # Un color por cada iteración
        for j in range(len(SD)):
            plt.scatter(longitudes_bases[SD[j]], latitudes_bases[SD[j]], color=color, label='Bases')
        plt.plot([longitudes_bases[SD[0]], longitudes_supply_depots[k]],[latitudes_bases[SD[0]], latitudes_supply_depots[k]], color='red')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()
