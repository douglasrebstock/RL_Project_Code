"""
RL 609 project experimental setup
Douglas Rebstock
"""

import math
from datetime import datetime

import pickle
from array import array

from environments import MountainCar
from learningAlgorithms import SemiGradSARSA, SemiGradSARSA_RANDO_GAMMA, SemiGradSARSA_MT_linear


class Experiment:

    def __init__(self, _algorithm, _environment, _timeOut, _numEps, _numRuns, _upscale=2):
        self.environment = _environment
        self.algorithm = _algorithm
        self.timeOut = _timeOut
        self.numEps = _numEps
        self.numRuns = _numRuns
        self.upscale = _upscale
        self.average_num_weights = 0

    def run(self):
        results = []
        for run in range(self.numRuns):
            running_average = 1000
            results.append(array("i",[0]*numEps))
            self.algorithm.reset_run()
            for episode in range(self.numEps):
                self.environment.reset()
                self.algorithm.reset_episode()
                state = self.environment.getState()
                features = self.algorithm.getFeatures(state)
                action, value = self.algorithm.getAction(features)
                self.algorithm.memoizeAction(action, value, features)
                terminal = self.environment.isTerminal()
                while not terminal:
                    state, reward, terminal = self.environment.increment(action)
                    self.algorithm.memoizeReward(reward)
                    features = self.algorithm.getFeatures(state)
                    action, value = self.algorithm.getAction(features)
                    self.algorithm.memoizeAction(action, value, features)
                    self.algorithm.update(terminal)
                    if self.environment.time > self.timeOut:
                        return None
                results[-1][episode] = int(self.environment.time)
                running_average *= 0.9
                running_average += 0.1 * self.environment.time
                self.average_num_weights += algorithm.tilingIHT.count()
                if episode % 9 == 0: print(running_average)
        self.average_num_weights /= self.numRuns
        return results

    def test(self, numEpsTest=10):
        average = 0
        for episode in range(numEpsTest):
            self.environment.reset()
            self.algorithm.reset_episode()
            state = self.environment.getState()
            features = self.algorithm.getFeatures(state)
            action, value = self.algorithm.getAction(features)
            terminal = self.environment.isTerminal()
            while not terminal:
                state, reward, terminal = self.environment.increment(action)
                features = self.algorithm.getFeatures(state)
                action, value = self.algorithm.getAction(features)
            average += self.environment.time
        average /= numEpsTest
        return average

# learning algorithm setup
lastN = 10
epsilon = 0
gamma = 1

# environment setup
posBounds = [-1.2, 0.5]
velBounds = [-0.07, 0.07]
startBound = [-0.6, -0.4]
timeIncrement = 1

# experimental settings
timeOut = 100000
numEps= 10000000
numRuns= 1
numTiles_grid = [5, 10, 15, 20]
numTilings_grid = [8, 16, 32, 64]
tau_grid = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
algorithms = [SemiGradSARSA]

results = {}
experimental_data = []
results["data"] = []
experiment_settings = {"numEps": numEps, "numRuns": numRuns}

for numTilings in numTilings_grid:
    for numTiles in numTiles_grid:
        numTiles_x = numTiles
        numTiles_v = numTiles
        x_scaleFactor = numTiles_x / (posBounds[1] - posBounds[0])
        v_scaleFactor = numTiles_v / (velBounds[1] - velBounds[0])
        maxSize = (numTiles_v + 1) * (numTiles_x + 1) * numTilings
        maxSize = int(math.pow(2, math.ceil(math.log2(maxSize))))
        for tau in tau_grid:
            for algo in algorithms:
                alpha = tau / numTilings
                algorithm = algo(alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize)
                name = algorithm.name
                environment = MountainCar(posBounds, velBounds, startBound)
                experiment = Experiment(algorithm, environment, timeOut, numEps, numRuns)
                results_train = experiment.run()
                if results_train is None:
                    continue
                avg = [0]*numEps
                for run in results_train:
                    for i in range(numEps):
                        avg[i] += run[i]
                for i in range(numEps): avg[i] /= numRuns
                std = [0]*numEps
                for run in results_train:
                    for i in range(numEps):
                        std[i] += math.pow((avg[i] - run[i]),2) / numRuns
                for i in range(numEps): std[i] = math.sqrt(std[i])
                data = {"numtilings": numTilings, "numTiles": numTiles, "alpha": alpha, "avg": avg, "std": std,
                        "algo": name, "rawdata": results_train, "num_weights": experiment.average_num_weights}
                experimental_data.append(data)


results = {"experimental_data": experimental_data, "experimental_settings": experiment_settings}

directory = "results/PARAMSTUDY/"
fileName = directory + "RESULTS"

with open(fileName, "wb") as results_file:
    pickle.dump(results, results_file)




