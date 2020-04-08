"""
episodic semi-gradient nstep Sarsa for estimating q hat (implementation of pseudocode pg 247 Sutton & Barto
Douglas Rebstock
"""

import math
import random
from tiles3 import tiles, IHT, hashcoords

from _collections import deque


class SemiGradSARSA:

    def __init__(self, alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize, modded=False):
        self.name = "SemiGradSARSA"
        self.alpha = alpha
        self.weights = {}
        self.errors = []
        self.epsilon = epsilon
        self.episode = []
        self.tilingIHT = IHT(maxSize)
        self.actions = [-1, 0, 1]
        self.numTilings = numTilings
        self.numActions = len(self.actions)
        self.lastN = lastN
        self.lastNActions = deque()
        self.lastNValues = deque()
        self.lastNRewards = deque()
        self.lastNFeatures = deque()
        self.timeStep = 0
        self.gamma = gamma
        self.modded = modded
        self.x_scaleFactor = x_scaleFactor
        self.v_scaleFactor = v_scaleFactor
        self.reset_run()


    def reset_run(self):
        for action in self.actions:
            self.weights[action] = [0] * self.tilingIHT.size
            self.errors = [0] * self.tilingIHT.size
            self.tilingIHT.reset()

    def upscale(self):
        return("supported in dynamic version")

    def reset_episode(self):
        self.lastNActions.clear()
        self.lastNValues.clear()
        self.lastNRewards.clear()
        self.lastNFeatures.clear()
        self.timeStep = 0

    def memoizeAction(self, action, value, features):
        self.lastNActions.append(action)
        self.lastNValues.append(value)
        self.lastNFeatures.append(tuple(features))
        if len(self.lastNActions) > self.lastN:
            self.lastNActions.popleft()
            self.lastNValues.popleft()
            self.lastNFeatures.popleft()


    def memoizeReward(self, reward):
        self.lastNRewards.append(reward)
        if len(self.lastNRewards) > self.lastN:
            self.lastNRewards.popleft()

    def update(self, terminal):
        reward = self.lastNRewards[-1]
        q_prime = self.lastNValues[-1]
        q = self.lastNValues[-2]
        action = self.lastNActions[-2]
        features = self.lastNFeatures[-2]
        if terminal:
            error = reward - q
        else:
            error = reward + self.gamma * q_prime - q
        for feature in features:
            self.weights[action][feature] += self.alpha * error

    def getAction(self, feature):
        actionVals = self.getActionValues(feature)
        # epsilon greedy action
        if random.random() < self.epsilon:
            index = random.randint(0, self.numActions - 1)
            return self.actions[index], actionVals[index]
        action = False
        # choose action with max value (randomizes checking order so won't bias)
        maxVal = -1e7
        randOrder = random.randint(0, self.numActions - 1)
        for i in range(len(self.actions)):
            index = (i + randOrder) % self.numActions
            if actionVals[index] > maxVal:
                maxVal = actionVals[index]
                action = self.actions[index]
        return action, maxVal

    def getFeatures(self, state):
        state[0] *= self.x_scaleFactor
        state[1] *= self.v_scaleFactor
        features, coords = tiles(self.tilingIHT, self.numTilings, state, [], False, self.modded)
        return features

    def getActionValues(self, features):
        actionValues = []
        for action in self.actions:
            actionValues.append(self.getActionValue(features,action))
        return actionValues

    def getActionValue(self, features, action):
        value = 0
        for tile in features:
            value += self.weights[action][tile]
        return value

    def getParameters(self):
        parameters = {}
        parameters["alpha"] = self.alpha
        parameters["x_scale"] = self.x_scaleFactor
        parameters["v_scale"] = self.v_scaleFactor
        parameters["epsilon"] = self.epsilon
        parameters["gamma"] = self.gamma
        return parameters


class SemiGradSARSA_DT(SemiGradSARSA):

    def __init__(self, alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize):
        self.maxSize_original = maxSize
        self.v_scaleFactor_original = v_scaleFactor
        self.x_scaleFactor_original = x_scaleFactor
        self.reset_weights = {}
        self.weights_old = {}
        self.tilingIHT_old = {}
        self.transferred = []
        self.transferCount = 0
        super().__init__(alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize)

    def reset_run(self):
        for action in self.actions:
            self.tilingIHT = IHT(self.maxSize_original)
            self.weights[action] = [0] * self.tilingIHT.size
            self.errors = [0] * self.tilingIHT.size
            self.x_scaleFactor = self.x_scaleFactor_original
            self.v_scaleFactor = self.v_scaleFactor_original
            self.transferred = [True] * self.maxSize_original
            self.weights_old.clear()

    def upscale(self):
        #print(self.transferCount)
        #print(self.tilingIHT.count())
        self.weights_old = self.weights.copy()
        self.tilingIHT_old = self.tilingIHT.copy()
        max_size = self.tilingIHT_old.size * 4
        self.tilingIHT = IHT(max_size)
        self.x_scaleFactor *= 2
        self.v_scaleFactor *= 2
        self.transferred = [False] * self.tilingIHT.size
        for action in self.actions:
            self.weights[action] = [0] * self.tilingIHT.size
            self.reset_weights[action] = [0] * self.tilingIHT.size
            self.errors = [0] * self.tilingIHT.size

    def getFeatures(self, state):
        state[0] *= self.x_scaleFactor
        state[1] *= self.v_scaleFactor
        features, coords = tiles(self.tilingIHT, self.numTilings, state, [], False)
        self.transfer_weights(features, coords)
        return features

    def transfer_weights(self, features, coords):
        for i in range(len(features)):
            feature = features[i]
            coord = coords[i]
            if not self.transferred[feature]:
                for j in range(3):
                    coord[j] = coord[j] // 2
                old_index = hashcoords(tuple(coord), self.tilingIHT_old, True)
                if old_index is not None:
                    self.transferCount += 1
                    for action in self.actions:
                        old_weight = self.weights_old[action][old_index]
                        self.weights[action][feature] = old_weight
                        self.reset_weights[action][feature] = old_weight
                self.transferred[feature] = True

class SemiGradSARSA_ANGLED_REG(SemiGradSARSA):

    def __init__(self, alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize):
        super().__init__(alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize)
        self.name = "SemiGradSARSA_ANGLED_REG"

    def getFeatures(self, state):
        features = []
        [x, v] = state[0], state[1]
        x *= self.x_scaleFactor
        v *= self.v_scaleFactor
        for i in range(self.numTilings):
            theta = (90/self.numTilings) * i
            q = x*math.cos(theta) - v*math.sin(theta)
            z = x*math.sin(theta) + v*math.cos(theta)
            feature_res , coords = tiles(self.tilingIHT, 1, [q, z], [i], False)
            features += feature_res
        return features

class SemiGradSARSA_RANDO(SemiGradSARSA):

    def __init__(self, alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize):
        self.tiling_trickery = []
        super().__init__(alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize)
        self.name = "SemiGradSARSA_RANDO"



    def getFeatures(self, state):
        features = []
        for i in range(self.numTilings):
            x = state[0] * self.x_scaleFactor * (self.tiling_trickery[i][0] + 0.5) + self.tiling_trickery[i][3]
            v = state[1] * self.v_scaleFactor * (self.tiling_trickery[i][1] + 0.5) + self.tiling_trickery[i][3]
            theta = 90 * self.tiling_trickery[i][2]
            q = x*math.cos(theta) - v*math.sin(theta)
            z = x*math.sin(theta) + v*math.cos(theta)
            feature_res , coords = tiles(self.tilingIHT, 1, [q, z], [i], False)
            features += feature_res
        return features

    def reset_run(self):
        for action in self.actions:
            self.weights[action] = [0] * self.tilingIHT.size
            self.errors = [0] * self.tilingIHT.size
            self.tilingIHT.reset()
            self.tiling_trickery.clear()
        for i in range(self.numTilings):
            self.tiling_trickery.append([random.random(), random.random(), random.random(), random.random()])

class SemiGradSARSA_RANDO_ANGLED(SemiGradSARSA):

    def __init__(self, alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize):
        self.tiling_trickery = []
        super().__init__(alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize)
        self.name = "SemiGradSARSA_RANDO_GAMMA"


    def getFeatures(self, state):
        features = []
        for i in range(self.numTilings):
            x = state[0] * (self.tiling_trickery[i][0]) + self.tiling_trickery[i][3]
            v = state[1] * (self.tiling_trickery[i][1]) + self.tiling_trickery[i][3]
            theta = 90 * self.tiling_trickery[i][2]
            q = x*math.cos(theta) - v*math.sin(theta)
            z = x*math.sin(theta) + v*math.cos(theta)
            feature_res , coords = tiles(self.tilingIHT, 1, [q, z], [i], False)
            features += feature_res
        return features

    def reset_run(self):
        for action in self.actions:
            self.weights[action] = [0] * self.tilingIHT.size
            self.errors = [0] * self.tilingIHT.size
            self.tilingIHT.reset()
            self.tiling_trickery.clear()
        random_list = []
        features_prop = 0
        for i in range(self.numTilings):
            random.gammavariate(7.5,1)
            random_list.append([random.gammavariate(7.5,1)*self.x_scaleFactor/7.5, random.gammavariate(7.5,1)*self.v_scaleFactor/7.5, random.random(), random.random()])
            if random_list[-1][0] > 4 * self.x_scaleFactor: random_list[-1][0] = 4 * self.x_scaleFactor
            if random_list[-1][1] > 4 * self.v_scaleFactor: random_list[-1][1] = 4 * self.v_scaleFactor
            features_prop += (random_list[-1][0] * random_list[-1][1])
        features_prop_orig = self.numTilings * self.x_scaleFactor * self.v_scaleFactor
        scale = math.sqrt(features_prop_orig/features_prop)
        for i in range(self.numTilings):
            random_list[i][0] *= scale
            random_list[i][1] *= scale
            self.tiling_trickery.append((random_list[i][0], random_list[i][1], random_list[i][2], random_list[i][3]))


class SemiGradSARSA_RANDO_GAMMA(SemiGradSARSA):

    def __init__(self, alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize):
        self.tiling_trickery = []
        super().__init__(alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize)
        self.name = "SemiGradSARSA_RANDO_GAMMA_NOTURN"


    def getFeatures(self, state):
        features = []
        for i in range(self.numTilings):
            x = state[0] * (self.tiling_trickery[i][0]) + self.tiling_trickery[i][3]
            v = state[1] * (self.tiling_trickery[i][1]) + self.tiling_trickery[i][3]
            feature_res , coords = tiles(self.tilingIHT, 1, [x, v], [i], False)
            features += feature_res
        return features

    def reset_run(self):
        for action in self.actions:
            self.weights[action] = [0] * self.tilingIHT.size
            self.errors = [0] * self.tilingIHT.size
            self.tilingIHT.reset()
            self.tiling_trickery.clear()
        random_list = []
        features_prop = 0
        for i in range(self.numTilings):
            random_list.append([random.gammavariate(7.5,1)*self.x_scaleFactor/7.5, random.gammavariate(7.5,1)*self.v_scaleFactor/7.5, random.random(), random.random()])
            if random_list[-1][0] > 4 * self.x_scaleFactor: random_list[-1][0] = 4 * self.x_scaleFactor
            if random_list[-1][1] > 4 * self.v_scaleFactor: random_list[-1][1] = 4 * self.v_scaleFactor

            features_prop += (random_list[-1][0] * random_list[-1][1])
        features_prop_orig = self.numTilings * self.x_scaleFactor * self.v_scaleFactor
        scale = math.sqrt(features_prop_orig/features_prop)
        for i in range(self.numTilings):
            random_list[i][0] *= scale
            random_list[i][1] *= scale
            self.tiling_trickery.append((random_list[i][0], random_list[i][1], random_list[i][2], random_list[i][3]))


class SemiGradSARSA_ANGLE(SemiGradSARSA):

    def __init__(self, alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize):
        super().__init__(alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize)
        self.name = "SemiGradSARSA_ANGLED"

    def getFeatures(self, state):
        features = []
        [x, v] = state[0], state[1]
        x *= self.x_scaleFactor
        v *= self.v_scaleFactor
        for i in range(self.numTilings):
            theta = (90/self.numTilings) * i
            q = (x + i/self.numTilings)*math.cos(theta) - (v + i/self.numTilings)*math.sin(theta)
            z = (x + i/self.numTilings)*math.sin(theta) + (v + i/self.numTilings)*math.cos(theta)
            feature_res , coords = tiles(self.tilingIHT, 1, [q, z], [i], False)
            features += feature_res
        return features

class SemiGradSARSA_MT(SemiGradSARSA):

    def __init__(self, alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize,numResolutions=2):
        self.numResolutions = numResolutions
        super().__init__(alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize)
        self.name = "SemiGradSARSA_MT"


    def getFeatures(self, state):
        features = []
        for i in range(self.numResolutions):
            x , v = state[0], state[1]
            x *= self.x_scaleFactor * math.pow(2,i)
            v *= self.v_scaleFactor * math.pow(2,i)
            numTiles = self.numTilings//self.numResolutions
            feature_res , coords = tiles(self.tilingIHT, numTiles, [x, v], [i], False)
            features += feature_res
        return features

class SemiGradSARSA_MT_linear(SemiGradSARSA):


    def __init__(self, alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize,
                 res_low=5,resExp=10):

        super().__init__(alpha, numTilings, epsilon, lastN, gamma, x_scaleFactor, v_scaleFactor, maxSize)
        self.name = "SemiGradSARSA_MT_linear"
        self.rescale = self.getRes(resExp,res_low)

    def getRes(self,resExp,res_low):
        numFeaturesExp = math.pow(resExp, 2) * self.numTilings
        numFeaturesExp = 8 * 200
        resExp = math.sqrt(numFeaturesExp/self.numTilings)
        res_high = math.sqrt(2 * numFeaturesExp / self.numTilings - math.pow(res_low, 2))
        res = []
        featureRange = [math.pow(res_low, 2), math.pow(res_high, 2)]
        for i in range(self.numTilings):
            area = (featureRange[1] - featureRange[0]) / (self.numTilings - 1) * i + featureRange[0]
            res.append(math.sqrt(area)/resExp)
        return res

    def getFeatures(self, state):
        features = []
        for i in range(self.numTilings):
            x = (i + state[0]) * self.x_scaleFactor * self.rescale[i]
            v = (3*i + state[1]) * self.v_scaleFactor * self.rescale[i]
            feature_res , coords = tiles(self.tilingIHT, 1, [x, v], [i], False)
            features += feature_res
        return features





