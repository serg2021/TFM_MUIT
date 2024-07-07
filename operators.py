import math
import random
import copy         #DESCOMENTAR PARA TIPOS DE RECURSOS

import numpy as np
import scipy as sp
import scipy.stats

def xorMask(vector, n, mode="byte"):
    """
    Applies an XOR operation between a random number and the input vector.
    """

    vector = vector.astype(int)
    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)

    if mode == "bin":
        mask = mask_pos
    elif mode == "byte":
        mask = np.random.randint(1, 0xFF, size=vector.shape) * mask_pos
    elif mode == "int":
        mask = np.random.randint(1, 0xFFFF, size=vector.shape) * mask_pos

    return vector ^ mask

def permutation(vector, n):
    """
    Randomly permutes 'n' of the components of the input vector.
    """

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)

    if np.count_nonzero(mask_pos == 1) < 2:
        mask_pos[random.sample(range(mask_pos.size), 2)] = 1

    shuffled_vec = vector[mask_pos]
    np.random.shuffle(shuffled_vec)
    vector[mask_pos] = shuffled_vec

    return vector

def mutate_noise(vector, params):
    """
    Adds random noise with a given probability distribution to 'n' components of the input vector.
    """

    method = params["method"]
    n = round(params["N"])

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)

    low = params.get("Low", -1)
    if isinstance(low, np.ndarray) and low.size == vector.size:
        low = low[mask_pos]

    up = params.get("Up", 1)
    if isinstance(up, np.ndarray) and up.size == vector.size:
        up = up[mask_pos]
    
    strength = params.get("F", 1)
    if isinstance(strength, np.ndarray) and strength.size == vector.size:
        strength = strength[mask_pos]

    avg = params.get("Avg", 0)
    if isinstance(avg, np.ndarray) and avg.size == vector.size:
        avg = avg[mask_pos]

    rand_vec = sampleDistribution(method, n, avg, strength, low, up)

    vector[mask_pos] = vector[mask_pos] + rand_vec
    return vector

def mutate_sample(vector, population, params):
    """
    Replaces 'n' components of the input vector with a random value sampled from a given probability distribution.
    """

    method = params["method"]
    n = round(params["N"])

    if isinstance(vector[0], list):
        mask_pos = np.hstack([np.ones(n), np.zeros(len(vector[0]) - n)]).astype(bool)

        low = params.get("Low", -1)
        if isinstance(low, np.ndarray) and low.size == vector.size:
            low = low[mask_pos]

        up = params.get("Up", 1)
        if isinstance(up, np.ndarray) and up.size == vector.size:
            up = up[mask_pos]
    
        strength = params.get("F", 1)
        if isinstance(strength, np.ndarray) and strength.size == vector.size:
            strength = strength[mask_pos]

        def media(popul):   #Hace la media para cada columna de la matriz, según indique mask_pos
            popul_matrix_A = [i.solution[0] for i in popul]
            popul_matrix_B = 1 * mask_pos
            popul_matrix_C = [i for i, v in enumerate(popul_matrix_B) if v != 0]
            media_out = []
            for i in popul_matrix_C:
                aux = 0
                contador = 0
                for j in range(len(popul)):
                    if isinstance(popul_matrix_A[j][i], list):
                        for k in range(len(popul_matrix_A[j][i])):
                            aux += popul_matrix_A[j][i][k]
                            contador += 1
                    else:
                        aux += popul_matrix_A[j][i]
                        contador += 1
                media_out.append(aux/contador)
            return media_out

        def desv(popul, med):
            popul_matrix_A = [i.solution[0] for i in popul]
            popul_matrix_B = 1 * mask_pos
            popul_matrix_C = [i for i, v in enumerate(popul_matrix_B) if v != 0]
            desv_out = []
            for i in popul_matrix_C:
                aux = 0
                contador = 0
                ind = 0
                for j in range(len(popul)):
                    if isinstance(popul_matrix_A[j][i], list):
                        for k in range(len(popul_matrix_A[j][i])):
                            aux += (popul_matrix_A[j][i][k] - med[ind])**2
                            contador += 1
                    else:
                        aux += (popul_matrix_A[j][i] - med[ind])**2
                        contador += 1
                aux2 = math.sqrt(aux/contador)
                desv_out.append(aux2)
                ind += 1
            return desv_out

        media_pop = np.array(media(population))
        desv_pop = np.array(desv(population, media_pop))

        rand_vec = sampleDistribution(method, n, media_pop, desv_pop, low, up)


        mask_pos_num = [i for i, v in enumerate(1*mask_pos) if v != 0]
        for i in mask_pos_num:
            ind_vec = 0
            if isinstance(vector[0][i], list):
                rand_val = random.choice(vector[0][i])
                ind_val = [j for j, v in enumerate(vector[1][i]) if v == rand_val]
                val = random.randint(0,29)
                for k in ind_val:
                    if int(rand_vec[ind_vec]) == 30:
                        vector[1][i][k] = val
                    else:
                        vector[1][i][k] = int(rand_vec[ind_vec])
                rand_ind = vector[0][i].index(rand_val)
                if int(rand_vec[ind_vec]) == 30:
                    vector[0][i][rand_ind] = val
                else:
                    vector[0][i][rand_ind] = int(rand_vec[ind_vec])
            else:
                ind_val = [j for j, v in enumerate(vector[1][i]) if v == vector[0][i]]
                val = random.randint(0,29)
                for k in ind_val:
                    if int(rand_vec[ind_vec]) == 30:
                        vector[1][i][k] = val
                    else:
                        vector[1][i][k] = int(rand_vec[ind_vec])
                if int(rand_vec[ind_vec]) == 30:
                    vector[0][i] = val
                else:
                    vector[0][i] = int(rand_vec[ind_vec])
            ind_vec += 1
        return vector
    else:
        # mask_pos = np.hstack([np.ones(n), np.zeros(vector.shape[1] - n)]).astype(bool)     #DESCOMENTAR PARA CADENA DE SUMINISTRO
        mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
        np.random.shuffle(mask_pos)

        mean = popul_matrix.mean(axis=0)[mask_pos]  #Hace la media para cada columna de la matriz, según indique mask_pos
        std = np.maximum(popul_matrix.std(axis=0)[mask_pos], 1e-6) * strength  # ensure there will be some standard deviation
        rand_vec = sampleDistribution(method, n, mean, std, low, up)

        #vector[0][mask_pos] = rand_vec     #DESCOMENTAR PARA CADENA DE SUMINISTRO
        vector[mask_pos] = rand_vec
        return vector

def rand_noise(vector, params):
    """
    Adds random noise with a given probability distribution to all components of the input vector.
    """

    method = params["method"]

    low = params.get("Low", -1)
    up = params.get("Up", 1)
    strength = params.get("F", 1)
    avg = params.get("Avg", 0)

    noise = sampleDistribution(method, vector.shape, avg, strength, low, up)

    return vector + noise

def rand_sample(vector, population, params):
    """
    Picks a vector with components sampled from a probability distribution.
    """

    method = params["method"]

    low = params.get("Low", -1)
    up = params.get("Up", 1)
    strength = params.get("F", 1)

    popul_matrix = np.vstack([i.solution for i in population])
    mean = popul_matrix.mean(axis=0)
    std = np.maximum(popul_matrix.std(axis=0), 1e-6) * strength  # ensure there will be some standard deviation

    rand_vec = sampleDistribution(method, vector.shape, mean, std, low, up)

    return rand_vec


"""
-Distribución zeta
-Distribución hipergeométrica
-Distribución geomética
-Distribución de Boltzman
-Distribución de Pascal (binomial negativa)
"""


def sampleDistribution(method, n, mean=0, strength=0.01, low=0, up=1):
    """
    Takes 'n' samples from a given probablility distribution and returns them as a vector.
    """

    sample = 0
    if method == "Gauss":
        sample = np.random.normal(mean, strength, size=n)
    elif method == "Uniform":
        sample = np.random.uniform(low, up, size=n)
    elif method == "Cauchy":
        sample = sp.stats.cauchy.rvs(mean, strength, size=n)
    elif method == "Laplace":
        sample = sp.stats.laplace.rvs(mean, strength, size=n)
    elif method == "Poisson":
        sample = sp.stats.poisson.rvs(strength, size=n)
    elif method == "Bernouli":
        sample = sp.stats.bernoulli.rvs(strength, size=n)
    else:
        print(f"Error: distribution \"{method}\" not defined")
        exit(1)
    return sample


def gaussian(vector, strength):
    """
    Adds random noise following a Gaussian distribution to the vector.
    """

    return rand_noise(vector, {"method": "Gauss", "F": strength})


def cauchy(vector, strength):
    """
    Adds random noise following a Cauchy distribution to the vector.
    """

    return rand_noise(vector, {"method": "Cauchy", "F": strength})


def laplace(vector, strength):
    """
    Adds random noise following a Laplace distribution to the vector.
    """

    return rand_noise(vector, {"method": "Laplace", "F": strength})


def uniform(vector, low, up):
    """
    Adds random noise following an Uniform distribution to the vector.
    """

    return rand_noise(vector, {"method": "Uniform", "Low": low, "Up": up})


def poisson(vector, mu):
    """
    Adds random noise following a Poisson distribution to the vector.
    """

    return rand_noise(vector, {"method": "Poisson", "F": mu})

"""
-Distribución zeta
-Distribución hipergeométrica
-Distribución geomética
-Distribución de Boltzman
-Distribución de Pascal (binomial negativa)
"""

def cross1p(vector1, vector2):
    """
    Performs a 1 point cross between two vectors.
    """

    cross_point = random.randrange(0, vector1.size)
    return np.hstack([vector1[:cross_point], vector2[cross_point:]])


def cross2p(vector1, vector2):
    """
    Performs a 2 point cross between two vectors.
    """

    cross_point1 = random.randrange(0, vector1.size - 2)
    cross_point2 = random.randrange(cross_point1, vector1.size)
    return np.hstack([vector1[:cross_point1], vector2[cross_point1:cross_point2], vector1[cross_point2:]])


def crossMp(vector1, vector2):
    """
    Performs a multipoint cross between two vectors.
    """

    if isinstance(vector1[0], list) and isinstance(vector2[0], list):
        mask_pos = 1 * (np.random.rand(len(vector1[0])) > 0.5)
        ind_mask = [i for i, v in enumerate(mask_pos) if v == 1]
        aux = copy.deepcopy(vector1)
        for j in ind_mask:
            aux[0][j] = copy.deepcopy(vector2[0][j])
            aux[1][j] = copy.deepcopy(vector2[1][j])
        return aux

    else:
        mask_pos = 1 * (np.random.rand(vector1.shape[1]) > 0.5)    #DESCOMENTAR PARA CADENA DE SUMINISTRO
        mask_pos = 1 * (np.random.rand(vector1.size) > 0.5)
        aux = np.copy(vector1)

        #aux[0][mask_pos == 1] = vector2[0][mask_pos == 1]          #DESCOMENTAR PARA CADENA DE SUMINISTRO
        aux[mask_pos == 1] = vector2[mask_pos == 1]
        return aux


def multiCross(vector, population, n_ind):
    """
    Performs a multipoint cross between the vector and 'n-1' individuals of the population
    """

    if n_ind >= len(population):
        n_ind = len(population)

    other_parents = random.sample(population, n_ind - 1)
    mask_pos = np.random.randint(n_ind, size=vector.shape[1]) - 1
    for i in range(0, n_ind - 1):
        vector[mask_pos == i] = other_parents[i].solution[mask_pos == i]
    return vector


def weightedAverage(vector1, vector2, alpha):
    """
    Performs a weighted average between the two given vectors
    """

    return alpha * vector1 + (1 - alpha) * vector2


def blxalpha(solution1, solution2, alpha):
    """
    Performs the BLX alpha crossing operator between two vectors.
    """
    
    alpha *= np.random.random()
    if isinstance(solution1, list) or isinstance(solution2, list):
        for i in range(len(solution1[0])):
            if isinstance(solution1[0][i], list) or isinstance(solution2[0][i], list):
                if isinstance(solution1[0][i], list):
                    for j in range(len(solution1[0][i])):
                        ind_val = [k for k, v in enumerate(solution1[1][i]) if v == solution1[0][i][j]] #Sacamos índices donde se repita un SD en la asignacion
                        val = random.randint(0, 29)
                        for l in ind_val:
                            if round(alpha * solution1[1][i][l]) == 30 or round(alpha * solution1[0][i][j]) == 30:
                                solution1[1][i][l] = val
                                solution1[0][i][j] = val
                            else:
                                solution1[1][i][l] = round(alpha * solution1[1][i][l])
                                solution1[0][i][j] = round(alpha * solution1[0][i][j])
                elif isinstance(solution2[0][i], list):
                    for j in range(len(solution2[0][i])):
                        ind_val = [k for k, v in enumerate(solution2[1][i]) if v == solution2[0][i][j]]
                        val = random.randint(0, 29)
                        for l in ind_val:
                            if round((1-alpha) * solution2[1][i][l]) == 30 or round((1-alpha) * solution2[0][i][j]) == 30:
                                solution2[1][i][l] = val
                                solution2[0][i][j] = val
                            else:
                                solution2[1][i][l] = round((1-alpha) * solution2[1][i][l])
                                solution2[0][i][j] = round((1-alpha) * solution2[0][i][j])
                else:
                    ind_val_1 = [k for k, v in enumerate(solution1[1][i]) if v == solution1[0][i]]
                    val = random.randint(0, 29)
                    for l in ind_val_1:
                        if round(alpha * solution1[1][i][l]) == 30:
                            solution1[1][i][l] = val
                        else:
                            solution1[1][i][l] = round(alpha * solution1[1][i][l])
                    if round(alpha * solution1[0][i]) == 30:
                        solution1[0][i] = val
                    else:
                        solution1[0][i] = round(alpha * solution1[0][i])
                    ind_val_2 = [k for k, v in enumerate(solution2[1][i]) if v == solution2[0][i]]
                    val = random.randint(0, 29)
                    for l in ind_val_2:
                        if round((1 - alpha) * solution2[1][i][l]) == 30:
                            solution2[1][i][l] = val
                        else:
                            solution2[1][i][l] = round((1 - alpha) * solution2[1][i][l])
                    if round((1 - alpha) * solution2[0][i]) == 30:
                        solution2[0][i] = val
                    else:
                        solution2[0][i] = round((1 - alpha) * solution2[0][i])

    #Una vez aplicado BLX tanto a las soluciones como a las asignaciones de clases a SD, las sumamos y obtenemos el coral definitivo

        solution3 = copy.deepcopy(solution1)

        for i in range(len(solution1[0])):
            if isinstance(solution1[0][i], list) and isinstance(solution2[0][i], list): #Sumo 2 listas
                ind_val_1 = [j for j,v in enumerate(solution1[1][i]) if v != 30]    #Sacamos las clases en cada base
                ind_val_2 = [j for j,v in enumerate(solution2[1][i]) if v != 30]    #Sacamos las clases en cada base
                contador = []
                for j in ind_val_1:
                    if j in ind_val_2:  #Comprobamos que comparten la misma clase para poder sumar
                        contador.append(j)
                if len(contador) > 0:
                    cant = random.randint(1, len(contador))   #Elegimos una cantidad aleatoria para sumar
                    vals = random.sample(contador,cant) #Elegimos qué índices sumaremos
                    for j in vals:
                        val_1 = solution1[1][i][j]
                        val_2 = solution2[1][i][j]
                        res = val_1 + val_2
                        if res == 30:
                            res = random.randint(0,29)
                        solution3[1][i][j] = res

                        #Ya tenemos actualizadas las clases. Ahora actualizamos la solución

                    solution3[0][i] = []
                    val_3 = [v for j, v in enumerate(solution3[1][i]) if v != 30]  # Sacamos las clases en cada base
                    for j in val_3:
                        solution3[0][i].append(j)
                    solution3[0][i] = list(set(solution3[0][i]))   #Si hemos repetido bases, nos quedamos con sólo una de cada
                    if len(solution3[0][i]) == 1:
                        solution3[0][i] = solution3[0][i][0]
                else:
                    continue

            if isinstance(solution1[0][i], list) or isinstance(solution2[0][i], list): #Sumo un entero y una lista
                ind_val_1 = [j for j,v in enumerate(solution1[1][i]) if v != 30]    #Sacamos las clases en cada base
                ind_val_2 = [j for j,v in enumerate(solution2[1][i]) if v != 30]    #Sacamos las clases en cada base
                contador = []
                for j in ind_val_1:
                    if j in ind_val_2:  #Comprobamos que comparten la misma clase para poder sumar
                        contador.append(j)
                if len(contador) > 0:
                    cant = random.randint(1, len(contador))  # Elegimos una cantidad aleatoria para sumar
                    vals = random.sample(contador, cant)  # Elegimos qué índices sumaremos
                    for j in vals:
                        val_1 = solution1[1][i][j]
                        val_2 = solution2[1][i][j]
                        res = val_1 + val_2
                        if res == 30:
                            res = random.randint(0, 29)
                        solution3[1][i][j] = res

                        # Ya tenemos actualizadas las clases. Ahora actualizamos la solución

                    if isinstance(solution3[0][i], list):
                        solution3[0][i] = []
                        val_3 = [v for j, v in enumerate(solution3[1][i]) if v != 30]  # Sacamos las clases en cada base
                        for j in val_3:
                            solution3[0][i].append(j)
                        solution3[0][i] = list(set(solution3[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                        if len(solution3[0][i]) == 1:
                            solution3[0][i] = solution3[0][i][0]
                    else:
                        val_3 = [v for j, v in enumerate(solution3[1][i]) if v != 30]  # Sacamos las clases en cada base
                        if len(val_3) == 1:
                            solution3[0][i] = val_3
                        else:
                            solution3[0][i] = []
                            val_3 = [v for j, v in enumerate(solution3[1][i]) if v != 30]  # Sacamos las clases en cada base
                            for j in val_3:
                                solution3[0][i].append(j)
                            solution3[0][i] = list(set(solution3[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                            if len(solution3[0][i]) == 1:
                                solution3[0][i] = solution3[0][i][0]
                else:
                    continue

            if isinstance(solution1[0][i], (int, np.integer)) and isinstance(solution2[0][i], (int, np.integer)): #Sumo un entero y otro entero
                ind_val_1 = [j for j, v in enumerate(solution1[1][i]) if v != 30]  # Sacamos las clases en cada base
                ind_val_2 = [j for j, v in enumerate(solution2[1][i]) if v != 30]  # Sacamos las clases en cada base
                contador = []
                for j in ind_val_1:
                    if j in ind_val_2:  # Comprobamos que comparten la misma clase para poder sumar
                        contador.append(j)
                if len(contador) > 0:
                    cant = random.randint(1, len(contador))  # Elegimos una cantidad aleatoria para sumar
                    vals = random.sample(contador, cant)  # Elegimos qué índices sumaremos
                    for j in vals:
                        val_1 = solution1[1][i][j]
                        val_2 = solution2[1][i][j]
                        res = val_1 + val_2
                        if res == 30:
                            res = random.randint(0,29)
                        solution3[1][i][j] = res

                        # Ya tenemos actualizadas las clases. Ahora actualizamos la solución

                    solution3[0][i] = []
                    val_3 = [v for j, v in enumerate(solution3[1][i]) if v != 30]  # Sacamos las clases en cada base
                    for j in val_3:
                        solution3[0][i].append(j)
                    solution3[0][i] = list(set(solution3[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                    if len(solution3[0][i]) == 1:
                        solution3[0][i] = solution3[0][i][0]
                else:
                    continue
        return solution3

    else:
        return alpha*solution1 + (1-alpha)*solution2


def sbx(solution1, solution2, strength):
    """
    Performs the SBX crossing operator between two vectors.
    """

    beta = np.zeros(solution1.shape)
    u = np.random.random(solution1.shape)
    for idx, val in enumerate(u):
        if val <= 0.5:
            beta[idx] = (2*val)**(1/(strength+1))
        else:
            beta[idx] = (0.5*(1-val))**(1/(strength+1))
    
    sign = random.choice([-1,1])
    return 0.5*(solution1+solution2) + sign*0.5*beta*(solution1-solution2)


def sim_annealing(solution, strength, objfunc, temp_changes, iter):
    """
    Performs a round of simulated annealing
    """

    p0, pf = (0.1, 7)

    alpha = 0.99
    best_fit = solution.get_fitness()

    temp_init = temp = 100
    temp_fin = alpha**temp_changes * temp_init
    while temp >= temp_fin:
        for j in range(iter):
            solution_new = gaussian(solution.solution, strength)
            new_fit = objfunc.fitness(solution_new)
            y = ((pf-p0)/(temp_fin-temp_init))*(temp-temp_init) + p0 

            p = np.exp(-y)
            if new_fit > best_fit or random.random() < p:
                best_fit = new_fit
        temp = temp*alpha
    return solution_new

def harmony_search(solution, population, strength, HMCR, PAR):
    """
    Performs the Harmony search operator
    """

    new_solution = np.zeros(solution.shape)
    popul_matrix = np.vstack([i.solution for i in population])
    popul_mean = popul_matrix.mean(axis=0)
    popul_std = popul_matrix.std(axis=0)
    for i in range(solution.size):
        if random.random() < HMCR:
            new_solution[i] = random.choice(population).solution[i]
            if random.random() <= PAR:
                new_solution[i] += np.random.normal(0,strength)
        else:
            new_solution[i] = np.random.normal(popul_mean[i], popul_std[i])
    return new_solution

def DERand1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/rand/1
    """

    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = r1.solution + F * (r2.solution - r3.solution)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DEBest1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/best/1
    """
    if isinstance(vector, list):
        if len(population) > 3:
            fitness = [i.fitness for i in population]
            best = population[fitness.index(max(fitness))]
            r1, r2 = random.sample(population, 2)
            for i in range(len(r1.solution[0])):
                if isinstance(r1.solution[0][i], list) or isinstance(r2.solution[0][i], list):
                    if isinstance(r1.solution[0][i], list):
                        for j in range(len(r1.solution[0][i])):
                            ind_val = [k for k, v in enumerate(r1.solution[1][i]) if v == r1.solution[0][i][j]]
                            val = random.randint(0,29)
                            for l in ind_val:
                                if round(F * r1.solution[1][i][l]) == 30:
                                    r1.solution[1][i][l] = val
                                else:
                                    r1.solution[1][i][l] = round(F * r1.solution[1][i][l])
                            if round(F * r1.solution[0][i][j]) == 30:
                                r1.solution[0][i][j] = val
                            else:
                                r1.solution[0][i][j] = round(F * r1.solution[0][i][j])
                    elif isinstance(r2.solution[0][i], list):
                        for j in range(len(r2.solution[0][i])):
                            ind_val = [k for k, v in enumerate(r2.solution[1][i]) if v == r2.solution[0][i][j]]
                            val = random.randint(0, 29)
                            for l in ind_val:
                                if round(F * r2.solution[1][i][l]) == 30:
                                    r2.solution[1][i][l] = val
                                else:
                                    r2.solution[1][i][l] = round(F * r2.solution[1][i][l])
                            if round(F * r2.solution[0][i][j]) == 30:
                                r2.solution[0][i][j] = val
                            else:
                                r2.solution[0][i][j] = round(F * r2.solution[0][i][j])
                    else:
                        ind_val_1 = [k for k, v in enumerate(r1.solution[1][i]) if v == r1.solution[0][i]]
                        val = random.randint(0, 29)
                        for l in ind_val:
                            if round(F * r1.solution[1][i][l]) == 30:
                                r1.solution[1][i][l] = val
                            else:
                                r1.solution[1][i][l] = round(F * r1.solution[1][i][l])
                        if round(F * r1.solution[0][i]) == 30:
                            r1.solution[0][i] = val
                        else:
                            r1.solution[0][i] = round(F * r1.solution[0][i])
                        ind_val_2 = [k for k, v in enumerate(r2.solution[1][i]) if v == r2.solution[0][i]]
                        for l in ind_val_2:
                            if round(F * r2.solution[1][i][l]) == 30:
                                r2.solution[1][i][l] = val
                            else:
                                r2.solution[1][i][l] = round(F * r2.solution[1][i][l])
                        if round(F * r2.solution[0][i]) == 30:
                            r2.solution[0][i] = val
                        else:
                            r2.solution[0][i] = round(F * r2.solution[0][i])

            #Ya hemos ponderado las soluciones multiplicando por F -> Ahora toca la resta y suma con best.solution

            solution_aux = copy.deepcopy(r1.solution)   #Hacemos una copia simplemente para mantener la estructura de listas de las soluciones

            for i in range(len(r1.solution[0])):
                if isinstance(r1.solution[0][i], list) and isinstance(r2.solution[0][i], list):  # Sumo 2 listas
                    ind_val_1 = [j for j, v in enumerate(r1.solution[1][i]) if v != 30]  # Sacamos las clases en cada base
                    ind_val_2 = [j for j, v in enumerate(r2.solution[1][i]) if v != 30]  # Sacamos las clases en cada base
                    contador = []
                    for j in ind_val_1:
                        if j in ind_val_2:  # Comprobamos que comparten la misma clase para poder sumar
                            contador.append(j)
                    if len(contador) > 0:
                        cant = random.randint(1, len(contador))  # Elegimos una cantidad aleatoria para sumar
                        vals = random.sample(contador, cant)  # Elegimos qué índices sumaremos
                        for j in vals:
                            val_1 = r1.solution[1][i][j]
                            val_2 = r2.solution[1][i][j]
                            res = val_1 - val_2
                            solution_aux[1][i][j] = res

                            # Ya tenemos actualizadas las clases. Ahora actualizamos la solución

                        solution_aux[0][i] = []
                        val_3 = [v for j, v in enumerate(solution_aux[1][i]) if v != 30]  # Sacamos las clases en cada base
                        for j in val_3:
                            solution_aux[0][i].append(j)
                        solution_aux[0][i] = list(set(solution_aux[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                        if len(solution_aux[0][i]) == 1:
                            solution_aux[0][i] = solution_aux[0][i][0]
                    else:
                        continue

                if isinstance(r1.solution[0][i], list) or isinstance(r2.solution[0][i], list):  # Sumo un entero y una lista
                    ind_val_1 = [j for j, v in enumerate(r1.solution[1][i]) if v != 30]  # Sacamos las clases en cada base
                    ind_val_2 = [j for j, v in enumerate(r2.solution[1][i]) if v != 30]  # Sacamos las clases en cada base
                    contador = []
                    for j in ind_val_1:
                        if j in ind_val_2:  # Comprobamos que comparten la misma clase para poder sumar
                            contador.append(j)
                    if len(contador) > 0:
                        cant = random.randint(1, len(contador))  # Elegimos una cantidad aleatoria para sumar
                        vals = random.sample(contador, cant)  # Elegimos qué índices sumaremos
                        for j in vals:
                            val_1 = r1.solution[1][i][j]
                            val_2 = r2.solution[1][i][j]
                            res = val_1 - val_2
                            solution_aux[1][i][j] = res

                            # Ya tenemos actualizadas las clases. Ahora actualizamos la solución

                        if isinstance(solution_aux[0][i], list):
                            solution_aux[0][i] = []
                            val_3 = [v for j, v in enumerate(solution_aux[1][i]) if v != 30]  # Sacamos las clases en cada base
                            for j in val_3:
                                solution_aux[0][i].append(j)
                            solution_aux[0][i] = list(set(solution_aux[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                            if len(solution_aux[0][i]) == 1:
                                solution_aux[0][i] = solution_aux[0][i][0]
                        else:
                            val_3 = [v for j, v in enumerate(solution_aux[1][i]) if v != 30]  # Sacamos las clases en cada base
                            if len(val_3) == 1:
                                solution_aux[0][i] = val_3
                            else:
                                solution_aux[0][i] = []
                                val_3 = [v for j, v in enumerate(solution_aux[1][i]) if v != 30]  # Sacamos las clases en cada base
                                for j in val_3:
                                    solution_aux[0][i].append(j)
                                solution_aux[0][i] = list(set(solution_aux[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                                if len(solution_aux[0][i]) == 1:
                                    solution_aux[0][i] = solution_aux[0][i][0]
                    else:
                        continue

                if isinstance(r1.solution[0][i], (int, np.integer)) and isinstance(r2.solution[0][i], (int, np.integer)):  # Sumo un entero y otro entero
                    ind_val_1 = [j for j, v in enumerate(r1.solution[1][i]) if v != 30]  # Sacamos las clases en cada base
                    ind_val_2 = [j for j, v in enumerate(r2.solution[1][i]) if v != 30]  # Sacamos las clases en cada base
                    contador = []
                    for j in ind_val_1:
                        if j in ind_val_2:  # Comprobamos que comparten la misma clase para poder sumar
                            contador.append(j)
                    if len(contador) > 0:
                        cant = random.randint(1, len(contador))  # Elegimos una cantidad aleatoria para sumar
                        vals = random.sample(contador, cant)  # Elegimos qué índices sumaremos
                        for j in vals:
                            val_1 = r1.solution[1][i][j]
                            val_2 = r2.solution[1][i][j]
                            res = val_1 - val_2
                            solution_aux[1][i][j] = res

                            # Ya tenemos actualizadas las clases. Ahora actualizamos la solución

                        solution_aux[0][i] = []
                        val_3 = [v for j, v in enumerate(solution_aux[1][i]) if v != 30]  # Sacamos las clases en cada base
                        for j in val_3:
                            solution_aux[0][i].append(j)
                        solution_aux[0][i] = list(set(solution_aux[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                        if len(solution_aux[0][i]) == 1:
                            solution_aux[0][i] = solution_aux[0][i][0]
                    else:
                        continue

            #A continuación toca sumar el resultado con best.solution

            val = copy.deepcopy(best.solution)

            for i in range(len(best.solution[0])):
                if isinstance(best.solution[0][i], list) and isinstance(solution_aux[0][i], list):  # Sumo 2 listas
                    ind_val_1 = [j for j, v in enumerate(best.solution[1][i]) if v != 30]  # Sacamos las clases en cada base
                    ind_val_2 = [j for j, v in enumerate(solution_aux[1][i]) if v != 30]  # Sacamos las clases en cada base
                    contador = []
                    for j in ind_val_1:
                        if j in ind_val_2:  # Comprobamos que comparten la misma clase para poder sumar
                            contador.append(j)
                    if len(contador) > 0:
                        cant = random.randint(1, len(contador))  # Elegimos una cantidad aleatoria para sumar
                        vals = random.sample(contador, cant)  # Elegimos qué índices sumaremos
                        for j in vals:
                            val_1 = best.solution[1][i][j]
                            val_2 = solution_aux[1][i][j]
                            res = val_1 + val_2
                            if res == 30:
                                res = random.randint(0,29)
                            val[1][i][j] = res

                            # Ya tenemos actualizadas las clases. Ahora actualizamos la solución

                        val[0][i] = []
                        val_3 = [v for j, v in enumerate(val[1][i]) if v != 30]  # Sacamos las clases en cada base
                        for j in val_3:
                            val[0][i].append(j)
                        val[0][i] = list(set(val[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                        if len(val[0][i]) == 1:
                            val[0][i] = val[0][i][0]
                    else:
                        continue

                if isinstance(best.solution[0][i], list) or isinstance(solution_aux[0][i], list):  # Sumo un entero y una lista
                    ind_val_1 = [j for j, v in enumerate(best.solution[1][i]) if v != 30]  # Sacamos las clases en cada base
                    ind_val_2 = [j for j, v in enumerate(solution_aux[1][i]) if v != 30]  # Sacamos las clases en cada base
                    contador = []
                    for j in ind_val_1:
                        if j in ind_val_2:  # Comprobamos que comparten la misma clase para poder sumar
                            contador.append(j)
                    if len(contador) > 0:
                        cant = random.randint(1, len(contador))  # Elegimos una cantidad aleatoria para sumar
                        vals = random.sample(contador, cant)  # Elegimos qué índices sumaremos
                        for j in vals:
                            val_1 = best.solution[1][i][j]
                            val_2 = solution_aux[1][i][j]
                            res = val_1 + val_2
                            if res == 30:
                                res = random.randint(0,29)
                            val[1][i][j] = res

                            # Ya tenemos actualizadas las clases. Ahora actualizamos la solución

                        if isinstance(val[0][i], list):
                            val[0][i] = []
                            val_3 = [v for j, v in enumerate(val[1][i]) if v != 30]  # Sacamos las clases en cada base
                            for j in val_3:
                                val[0][i].append(j)
                            val[0][i] = list(set(val[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                            if len(val[0][i]) == 1:
                                val[0][i] = val[0][i][0]
                        else:
                            val_3 = [v for j, v in enumerate(val[1][i]) if v != 30]  # Sacamos las clases en cada base
                            if len(val_3) == 1:
                                val[0][i] = val_3
                            else:
                                val[0][i] = []
                                val_3 = [v for j, v in enumerate(val[1][i]) if v != 30]  # Sacamos las clases en cada base
                                for j in val_3:
                                    val[0][i].append(j)
                                val[0][i] = list(set(val[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                                if len(val[0][i]) == 1:
                                    val[0][i] = val[0][i][0]
                    else:
                        continue

                if isinstance(best.solution[0][i], (int, np.integer)) and isinstance(solution_aux[0][i], (int, np.integer)):  # Sumo un entero y otro entero
                    ind_val_1 = [j for j, v in enumerate(best.solution[1][i]) if v != 30]  # Sacamos las clases en cada base
                    ind_val_2 = [j for j, v in enumerate(solution_aux[1][i]) if v != 30]  # Sacamos las clases en cada base
                    contador = []
                    for j in ind_val_1:
                        if j in ind_val_2:  # Comprobamos que comparten la misma clase para poder sumar
                            contador.append(j)
                    if len(contador) > 0:
                        cant = random.randint(1, len(contador))  # Elegimos una cantidad aleatoria para sumar
                        vals = random.sample(contador, cant)  # Elegimos qué índices sumaremos
                        for j in vals:
                            val_1 = best.solution[1][i][j]
                            val_2 = solution_aux[1][i][j]
                            res = val_1 + val_2
                            if res == 30:
                                res = random.randint(0,29)
                            val[1][i][j] = res

                            # Ya tenemos actualizadas las clases. Ahora actualizamos la solución

                        val[0][i] = []
                        val_3 = [v for j, v in enumerate(val[1][i]) if v != 30]  # Sacamos las clases en cada base
                        for j in val_3:
                            val[0][i].append(j)
                        val[0][i] = list(set(val[0][i]))  # Si hemos repetido bases, nos quedamos con sólo una de cada
                        if len(val[0][i]) == 1:
                            val[0][i] = val[0][i][0]
                    else:
                        continue

            mask_pos = 1 * (np.random.random(len(vector[0])) <= CR)
            ind_mask = [i for i, v in enumerate(mask_pos) if v == 1]
            for j in ind_mask:
                vector[0][j] = copy.deepcopy(val[0][j])
                vector[1][j] = copy.deepcopy(val[1][j])
            return vector
        else:
            for i in range(len(vector[0])): #Aseguramos que no se repitan SD en la solución
                if isinstance(vector[0][i], list):
                    vector[0][i] = list(set(vector[0][i]))
            return vector

        #FALLA AQUÍ, PORQUE CUANDO NO DETECTA EL IF, NO TIENE NADA QUE DEVOLVER Y DEVUELVE NONE
        #Mañana hacemos la reparación, que consistirá en un else con la solución de entrada de vuelta (corregida por si acaso)


    else:
        if len(population) > 3:
            fitness = [i.fitness for i in population]
            best = population[fitness.index(max(fitness))]
            r1, r2 = random.sample(population, 2)

            v = best.solution + F * (r1.solution - r2.solution)
            #mask_pos = np.random.random(vector.shape[1]) <= CR     #DESCOMENTAR PARA CADENA DE SUMINISTRO
            #vector[0][mask_pos] = v[0][mask_pos]                   #DESCOMENTAR PARA CADENA DE SUMINISTRO
            mask_pos = np.random.random(vector.shape) <= CR
            vector[mask_pos] = v[mask_pos]
        return vector


def DERand2(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/rand/2
    """

    if len(population) > 5:
        r1, r2, r3, r4, r5 = random.sample(population, 5)

        v = r1.solution + F * (r2.solution - r3.solution) + F * (r4.solution - r5.solution)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DEBest2(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/best/2
    """

    if len(population) > 5:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2, r3, r4 = random.sample(population, 4)

        v = best.solution + F * (r1.solution - r2.solution) + F * (r3.solution - r4.solution)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DECurrentToRand1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/current-to-rand/1
    """

    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = vector + np.random.random() * (r1.solution - vector) + F * (r2.solution - r3.solution)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DECurrentToBest1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/current-to-best/1
    """

    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = vector + F * (best.solution - vector) + F * (r1.solution - r2.solution)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DECurrentToPBest1(vector, population, F, CR, P):
    """
    Performs the differential evolution operator DE/current-to-pbest/1
    """

    if len(population) > 3:
        fitness = [i.fitness for i in population]
        upper_idx = max(1, math.ceil(len(population) * P))
        pbest_idx = random.choice(np.argsort(fitness)[:upper_idx])
        pbest = population[pbest_idx]
        r1, r2 = random.sample(population, 2)

        v = vector + F * (pbest.solution - vector) + F * (r1.solution - r2.solution)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector

def firefly(solution, population, objfunc, alpha_0, beta_0, delta, gamma):
    sol_range = objfunc.sup_lim - objfunc.inf_lim
    n_dim = solution.solution.size
    new_solution = solution.solution.copy()
    for idx, ind in enumerate(population):
        if solution.get_fitness() < ind.get_fitness():
            r = np.linalg.norm(solution.solution - ind.solution)
            alpha = alpha_0 * delta ** idx
            beta = beta_0 * np.exp(-gamma*(r/(sol_range*np.sqrt(n_dim)))**2)
            new_solution = new_solution + beta*(ind.solution-new_solution) + alpha * sol_range * random.random()-0.5
            new_solution = objfunc.repair_solution(new_solution)
    
    return new_solution

def dummy_op(vector, scale=1000):
    """
    Replaces the vector with one consisting of all the same value

    Only for testing, not useful for real applications
    """

    return np.ones(vector.shape) * scale
