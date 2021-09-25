import json
from random import random as rand
from math import e as euler
import os
import asyncio
import sys
class NeuralNetwork:
    def __init__(self,structure):
        self.sumError = float()
        self.network = []
        self.trained = False
        self.iterationNumber = 0
        self.structure = structure
        for layerNumber, layer in enumerate(structure[1:]):
            newLayer = []
            for neuronNumber in range(layer):
                neuron = dict()
                
                if layerNumber != len(structure)-1:neuron['weights'] = [rand() for x in range(structure[layerNumber-1]+1)]
                else:neuron['weights'] = [rand() for x in range(structure[layerNumber-1])]
                neuron['error'] = float()
                neuron['output'] = float()
                newLayer.append(neuron)
            self.network.append(newLayer)
    def importNetwork(self, directory, **kwargs):
        json.load(open(f"{directory}.neural",'r'))
    def exportNetwork(self, directory, **kwargs):
        with open(f'{directory}.neural','w') as file:
            json.dump(self.network, file)
            file.close()
    def predict(self, data, **kwargs):
        print_out = False
        if 'print_out' in kwargs:
            print_out = True
        inputs = data
        newInputs = []
        for layerNumber, layer in enumerate(self.network[1:]):
            for neuronNumber, neuron in enumerate(layer):
                for weightNumber, weight in enumerate(neuron['weights'][:-1]):
                    neuron['output'] += weight*inputs[weightNumber]
                neuron['output'] = 1/(1+(euler**(-float(neuron['output']))))
                newInputs.append(neuron['output'])
            inputs = newInputs
            newInputs = []
        outputs = inputs
        if print_out:
            print(outputs)
        return outputs
    def trainNetwork(self, iterations, learnRate, dataset, **kwargs):
        previousError = 50000000
        print_out = False
        if 'noOut' in kwargs:
            if kwargs['noOut']:
                print_out = False
            else:
                print_out = True
        for iteration in range(iterations):
            self.sumError = 0
            self.iterationNumber += 1
            for dataNumber, data in enumerate(dataset):
                inputs = data[:-1]
                for inputValue in inputs:
                    inputValue = float(inputValue)
                newInputs = []
                for layerNumber, layer in enumerate(self.network[1:]):
                    for neuronNumber, neuron in enumerate(layer):
                        neuron['output'] = neuron['weights'][-1]
                        for weightNumber, weight in enumerate(neuron['weights'][:-1]):
                            neuron['output'] += weight*inputs[weightNumber]
                        neuron['output'] = 1/(1+(euler**(-float(neuron['output']))))
                        newInputs.append(neuron['output'])
                    inputs = newInputs
                    newInputs = []
                outputs = inputs
                expected = data[-1]
                for expectedNumber, expectedValue in enumerate(expected):
                    self.sumError += (expectedValue-outputs[expectedNumber])**2
                ## Error backward propagation
                for layerNumber in reversed(range(len(self.network))):
                    layer = self.network[layerNumber]
                    if layerNumber != len(self.network)-1:
                        for neuronNumber in range(len(layer)):
                            for neuron in self.network[layerNumber+1]:
                                self.network[layerNumber][neuronNumber]['error'] += neuron['weights'][neuronNumber]*neuron['error']
                            self.network[layerNumber][neuronNumber]['error'] = self.network[layerNumber][neuronNumber]['error']*(self.network[layerNumber][neuronNumber]['output']*(1-self.network[layerNumber][neuronNumber]['output']))
                    else:
                        for neuronNumber in range(len(layer)):
                            neuron = layer[neuronNumber]
                            neuron['error'] = (expected[neuronNumber]-neuron['output'])*(neuron['output']*(1-neuron['output']))
                ## Updates the weights
                for layerNumber in range(len(self.network)):
                    inputs = []
                    if layerNumber != 0:
                        for neuron in self.network[layerNumber-1]:
                            inputs.append(neuron['output'])
                    else:
                        inputs = data
                    layer = self.network[layerNumber]
                    for neuronNumber in range(len(layer)):
                        neuron = layer[neuronNumber]
                        for inputNumber in range(len(inputs)-1):
                            neuron['weights'][inputNumber] = float(neuron['weights'][inputNumber]) + (float(learnRate)*float(neuron['error'])*float(inputs[inputNumber]))
                        neuron['weights'][-1] += neuron['error']*learnRate
            if print_out:
                print(f'Iteration: {self.iterationNumber}\t Error: {self.sumError}')
    def getError(self):
        return self.sumError

network = NeuralNetwork([2,3,3,2])
print(network.network)
network.trainNetwork(100000, 0.3, [[0,0,[1,0]],[1,1,[0,1]],[0,1,[1,0]],[1,0,[1,0]]], noOut = False)
output = network.predict([1,1])
print(output.index(max(output)))