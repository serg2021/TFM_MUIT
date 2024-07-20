# -*- coding: utf-8 -*-
"""
@author: Sergio

1º -> Activamos el entorno de conda -> conda activate TFM_MUIT
2º -> Podemos ejecutar el código sin problemas
"""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
import csv
from sklearn.cluster import KMeans

def Puntos_Sin_Repetir(num_points, offset=0.5):
    with rasterio.open(mapa_dem) as dem:
        limites = dem.bounds
    points = set()  # Usamos un conjunto para evitar duplicados
    while len(points) < num_points:
        latitud = np.random.uniform(low=limites.bottom, high=limites.top)
        longitud = np.random.uniform(low=limites.left, high=limites.right)
        # Aplicar desplazamiento aleatorio para evitar superposiciones
        latitud_offset = np.random.uniform(low=-offset, high=offset)
        longitud_offset = np.random.uniform(low=-offset, high=offset)
        point_with_offset = (latitud + latitud_offset, longitud + longitud_offset)
        points.add(point_with_offset)  # Agregamos el punto al conjunto
    return points

if __name__ == "__main__":
    # Definicion de los parámetros del genético

    mapa_dem = 'PNOA_MDT05_ETRS89_HU30_0560_LID.tif'
    numero_bases = 200
    numero_supply_depots = 10
    capacidad_maxima = 20
    Ruta_Puntos = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Despliegue_Optimo',
        f"Bases_SD.csv")
    Ruta_Capacidades = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Despliegue_Optimo',
        f"Cap_Bases_SD.csv")
    if not os.path.exists(Ruta_Puntos):
        puntos = list(Puntos_Sin_Repetir(numero_bases))
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
        puntos = np.array(puntos)
    bases = puntos[:numero_bases]
    longitudes_bases, latitudes_bases = zip(*bases)
    if not os.path.exists(Ruta_Capacidades):
        capacidad_bases = np.random.randint(1, capacidad_maxima, size=(numero_bases))
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
    capacidad_supply_depots = np.full(numero_supply_depots,200)
       ### A CONTINUACIÓN, APLICAMOS EL CLUSTERING

    centroides = KMeans(n_clusters=numero_supply_depots).fit(bases)
    supply_depots = centroides.cluster_centers_
    longitudes_supply_depots, latitudes_supply_depots = zip(*supply_depots)

    puntos_def = np.vstack((bases, supply_depots))
    Ruta_Escenario = os.path.join(
        r'C:\Users\sergi\OneDrive - Universidad de Alcala\Escritorio\Universidad_Sergio\Master_Teleco\TFM\TFM_MUIT\Despliegue_Optimo',
        f"Escenario_Optimo.csv")
    np.savetxt(Ruta_Escenario, puntos_def, delimiter=',')
    # Graficar el mapa y los puntos
    fig_1 = plt.figure(figsize=(10, 6))
    plt.scatter(longitudes_bases, latitudes_bases, color='blue', label='Bases')
    plt.scatter(longitudes_supply_depots, latitudes_supply_depots, color='black', marker='p',label='Puntos de Suministro')
    fig_1.show()
