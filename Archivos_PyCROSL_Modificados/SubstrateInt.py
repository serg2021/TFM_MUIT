import random
from copy import copy
import numpy as np
from PyCROSL.operators import *
from PyCROSL.Substrate import *

numero_supply_depots = 10       #PONER VALOR PARA TIPOS DE RECURSOS B

class SubstrateInt(Substrate):
    """
    Substrate class that has discrete mutation and cross methods
    """

    def __init__(self, evolution_method, params = None):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method, params)
    
    def evolve(self, solution, population, objfunc):
        """
        Applies a mutation method depending on the type of operator
        """

        others = [i for i in population if i != solution]
        if len(others) == 0:
            solution2 = solution
            others = [solution2]
        else:
            solution2 = random.choice(others)

        params = copy.deepcopy(self.params)

        if "Cr" in params and "N" not in params:
            params["N"] = np.count_nonzero(np.random.random(solution.solution.size) < params["Cr"])    #DESCOMENTAR PARA EL RESTO
            #params["N"] = np.count_nonzero(np.random.random(len(solution.solution[0])) < params["Cr"])  #DESCOMENTAR PARA TIPOS DE RECURSOS

        if "N" in params:
            params["N"] = round(params["N"])
        
        if self.evolution_method == "1point":
            result = cross1p(solution.solution.copy(), solution2.solution.copy())
        elif self.evolution_method == "2point":
            result = cross2p(solution.solution.copy(), solution2.solution.copy())
        elif self.evolution_method == "Multipoint":
            result = crossMp(solution.solution.copy(), solution2.solution.copy())
        elif self.evolution_method == "WeightedAvg":
            result = weightedAverage(solution.solution.copy(), solution2.solution.copy(), params["F"])
        elif self.evolution_method == "BLXalpha":
            result = blxalpha(solution.solution.copy(), solution2.solution.copy(), params["F"])
        elif self.evolution_method == "Multicross":
            result = multiCross(solution.solution.copy(), others, params["N"])
        elif self.evolution_method == "Perm":
            result = permutation(solution.solution.copy(), params["N"])
        elif self.evolution_method == "Xor":
            result = xorMask(solution.solution.copy(), params["N"])
        elif self.evolution_method == "MutNoise":
            result = mutate_noise(solution.solution.copy(), params)
        elif self.evolution_method == "MutSample":
            result = mutate_sample(solution.solution.copy(), population, params)
        elif self.evolution_method == "RandNoise":
            result = rand_noise(solution.solution.copy(), params)
        elif self.evolution_method == "RandSample":
            result = rand_sample(solution.solution.copy(), population, params)
        elif self.evolution_method == "Gauss":
            result = gaussian(solution.solution.copy(), params["F"])
        elif self.evolution_method == "Laplace":
            result = laplace(solution.solution.copy(), params["F"])
        elif self.evolution_method == "Cauchy":
            result = cauchy(solution.solution.copy(), params["F"])
        elif self.evolution_method == "Uniform":
            result = uniform(solution.solution.copy(), params["Low"], params["Up"])
        elif self.evolution_method == "Poisson":
            result = poisson(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "DE/rand/1":
            result = DERand1(solution.solution.copy(), others, params["F"], params["Cr"])
        elif self.evolution_method == "DE/best/1":
            result = DEBest1(solution.solution.copy(), others, params["F"], params["Cr"])
        elif self.evolution_method == "DE/rand/2":
            result = DERand2(solution.solution.copy(), others, params["F"], params["Cr"])
        elif self.evolution_method == "DE/best/2":
            result = DEBest2(solution.solution.copy(), others, params["F"], params["Cr"])
        elif self.evolution_method == "DE/current-to-rand/1":
            result = DECurrentToRand1(solution.solution.copy(), others, params["F"], params["Cr"])
        elif self.evolution_method == "DE/current-to-best/1":
            result = DECurrentToBest1(solution.solution.copy(), others, params["F"], params["Cr"])
        elif self.evolution_method == "DE/current-to-pbest/1":
            result = DECurrentToPBest1(solution.solution.copy(), others, params["F"], params["Cr"], params["P"])
        elif self.evolution_method == "LSHADE":
            params["Cr"] = np.random.normal(params["Cr"], 0.1)
            params["F"] = np.random.normal(params["F"], 0.1)

            params["Cr"] = np.clip(params["Cr"], 0, 1)
            params["F"] = np.clip(params["F"], 0, 1)

            result = DECurrentToPBest1(solution.solution.copy(), others, params["F"], params["Cr"], params["P"])            
        elif self.evolution_method == "SA":
            result = sim_annealing(solution, params["F"], objfunc, params["temp_ch"], params["iter"])
        elif self.evolution_method == "HS":
            result = harmony_search(solution.solution.copy(), population, params["F"], params["Cr"], params["Par"])
        elif self.evolution_method == "Replace":
            result = replace(solution.solution.copy(), population, params["method"], params["F"])
        elif self.evolution_method == "Dummy":
            result = dummy_op(solution.solution.copy(), params["F"])
        elif self.evolution_method == "Custom":
            fn = params["function"]
            result = fn(solution.solution.copy(), population, objfunc, params)
        else:
            print(f"Error: evolution method \"{self.evolution_method}\" not defined")
            exit(1)

        if isinstance(result, list):
            for i in range(len(result[0])):
                if isinstance(result[0][i], list):
                    result[0][i] = list(set(result[0][i]))
                    for j in range(len(result[0][i])):
                        if result[0][i][j] > numero_supply_depots-1 or result[0][i][j] < 0:
                            ind_asig = [l for l, v in enumerate(result[1][i]) if v == result[0][i][j]]
                            result[0][i][j] = np.random.randint(0,numero_supply_depots)
                            for k in ind_asig:
                                result[1][i][k] = result[0][i][j]  # Actualizamos la asignación de clases de la base mutada
                    if len(result[0][i]) == 1:
                        result[0][i] = result[0][i][0]
                elif result[0][i] > numero_supply_depots-1 or result[0][i] < 0:
                    ind_asig = [j for j, v in enumerate(result[1][i]) if v == result[0][i]]
                    result[0][i] = np.random.randint(0, numero_supply_depots)
                    for k in ind_asig:
                        result[1][i][k] = result[0][i]  # Actualizamos la asignación de clases de la base mutada

            return result
        else:
            return np.round(result)