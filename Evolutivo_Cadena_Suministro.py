# -*- coding: utf-8 -*-
"""
@author: Sergio

1º -> Activamos el entorno de conda -> conda activate TFM_MUIT
2º -> Podemos ejecutar el código sin problemas
"""
import numpy as np
from PyCROSL.CRO_SL import CRO_SL
import random
import matplotlib.pyplot as plt

class EvolutiveClass:
    def __init__(self, Num_Individuos=200, Num_Generaciones=10, Tam_Individuos=10, Num_Max = 5, Prob_Padres=0.5, Prob_Mutacion=0.02, Prob_Cruce=0.5):
        self.Num_Individuos = Num_Individuos
        self.Num_Generaciones = Num_Generaciones
        self.Tam_Individuos = Tam_Individuos
        self.Num_Max = Num_Max
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
    
    def PoblacionInicial(self, Fil=None, Col=None, Num_Max=None):
        if Fil == None:
            Fil = self.Num_Individuos
        if Col == None:
            Col = self.Tam_Individuos
        if Num_Max == None:
            Num_Max = self.Num_Max
        Pob_Ini = np.random.randint(0,Num_Max, size=(Fil,Col))
        return Pob_Ini

    def Seleccion(self, poblacion_inicial, coste):
        index = np.argsort(coste)
        coste_ordenado = np.sort(coste)
        poblacion_actual = poblacion_inicial[index,:]
        poblacion_actual = poblacion_actual[0:self.Num_Padres,:]
        return poblacion_actual, coste_ordenado

    def Cruce (self, poblacion, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        for i in range (self.Num_Individuos - self.Num_Padres):
            Indice_Padres = random.sample(range(self.Num_Padres), 2)            # Se elige aleatoriamente el indice de los padres
            Padre1 = poblacion[Indice_Padres[0],:]                              # Se coge el padre 1
            Padre2 = poblacion[Indice_Padres[1],:]                              # Se coge el padre 2
            Hijo = np.copy(Padre1)                                              # El hijo va a ser una copia del padre 1
            vector = 1*(np.random.rand(self.Tam_Individuos) > self.Prob_Cruce)  # Se genera un vector para seleccionar los genes del padre 2
            Hijo[np.where(vector==1)[0]] = Padre2[np.where(vector==1)[0]]       # Los genes seleccionados del padre 2 pasan al hijo
            if np.random.rand() < self.Prob_Mutacion:                           # Se comprueba si el hijo va a mutar
                Hijo = self.Mutacion(Hijo, Num_Max)
            # if sum(Hijo) != self.Max_BBU:                                       # Se comprueba si hay que reparar el hijo
            #     Hijo = self.Reparacion (Hijo, Lista_Fibra)
            poblacion = np.insert(poblacion,self.Num_Padres+i,Hijo, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado
        return poblacion

    def Mutacion (self, individuo, Num_Max=None):                                
        aux1 = np.random.randint(0, individuo.shape[0])                         # Se genera número aleatorio para ver la posición que muta
        aux2 = np.random.randint(0,Num_Max)                                   # Se genera el número a modificar
        individuo[aux1] = aux2        
        return individuo

def Puntos_Sin_Repetir(num_points, offset=0.2):
    points = set()  # Usamos un conjunto para evitar duplicados
    while len(points) < num_points:
        latitud = np.random.uniform(low=-90.0, high=90.0)
        longitud = np.random.uniform(low=-180.0, high=180.0)
        points.add((latitud, longitud))  # Agregamos el punto al conjunto
        # Aplicar desplazamiento aleatorio para evitar superposiciones
        latitud_offset = np.random.uniform(low=-offset, high=offset)
        longitud_offset = np.random.uniform(low=-offset, high=offset)
        point_with_offset = (latitud + latitud_offset, longitud + longitud_offset)
        points.add(point_with_offset)  # Agregamos el punto al conjunto
    return points

if __name__ == "__main__":
    # Definicion de los parámetros del genético
    Num_Individuos = 200
    Num_Generaciones = 20
    Tam_Individuos = 100
    Prob_Padres = 0.1
    Prob_Mutacion = 0.01
    Prob_Cruce = 0.5

    numero_bases = 200
    bases = Puntos_Sin_Repetir(numero_bases)
    longitudes_bases, latitudes_bases = zip(*bases)

    numero_supply_depots = 20
    supply_depots = Puntos_Sin_Repetir(numero_supply_depots)
    longitudes_supply_depots, latitudes_supply_depots = zip(*supply_depots)

    # Graficar el mapa y los puntos
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p', label='Puntos de Suministro')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa con Puntos Aleatorios')
    plt.legend(bbox_to_anchor=(0, 0), loc='upper left')
    plt.show()
    
    #Ev1 = EvolutiveClass(Num_Individuos, Num_Generaciones, Tam_Individuos, Prob_Padres, Prob_Mutacion, Prob_Cruce)
    #Ev1.ImprimirInformacion()
    #Pob = Ev1.PoblacionInicial(None, 100, 5)