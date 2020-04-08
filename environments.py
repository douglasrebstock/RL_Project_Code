"""
Mountain car implementation from Sutton & Barto
Douglas Rebstock
"""

import math
import random

class physicalValue:

    def __init__(self, bound, value):
        self.bounds = bound
        self.value = value

    def bound(self):
        if self.value < self.bounds[0]:
            self.value = self.bounds[0]
        elif self.value > self.bounds[1]:
            self.value = self.bounds[1]

class MountainCar:

    def __init__(self, posBounds, velBounds, startBound, timeIncrement = 1):
        self.posBounds = posBounds
        self.startBound = startBound
        self.x = physicalValue(posBounds, 0)
        self.v = physicalValue(velBounds, 0)
        self.timeIncrement = timeIncrement
        self.time = 0
        self.reset()

    def reset(self):
        self.x.value = random.random() * (self.startBound[1] - self.startBound[0]) + self.startBound[0]
        self.v.value = 0
        self.time = 0

    def getState(self):
        state = [self.x.value, self.v.value]
        return state

    def increment(self, action):
        self.v.value = self.v.value + 0.001 * action - 0.0025 * math.cos(3*self.x.value)
        self.v.bound()
        self.x.value = self.x.value + self.v.value * self.timeIncrement
        self.x.bound()
        if self.x.value == self.posBounds[0]:
            self.v.value = 0
        state = [self.x.value, self.v.value]
        terminated = self.isTerminal()
        if terminated:
            reward = 0
        else:
            reward = -1
        self.time += 1
        return state, reward, terminated

    def isTerminal(self):
        return self.x.value == self.posBounds[1]

    def getParameters(self):
        parameters = {}
        parameters["posBounds"] = self.x.bounds
        parameters["velBounds"] = self.v.bounds
        parameters["startBound"] = self.startBound
        parameters["timeIncrement"] = self.timeIncrement
        return parameters














