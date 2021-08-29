import json
from random import random as rand
from math import euler
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
                neuron['weights'] = [rand() for x in range(structure[layerNumber-1]+1)]
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
        if 'print_out' in kwargs:
            print_out = True
        inputs = data
        newInputs = []
        for layerNumber, layer in enumerate(self.network[1:]):
            for neuronNumber, neuron in enumerate(layer):
                for weightNumber, weight in enumerate(neuron['weights']):
                    neuron['output'] += weight*inputs[weightNumber]
                    neuron['output'] = 1/(1+(math.e**(-float(neuron['output']))))
                    newInputs.append(neuron['output'])
            inputs = newInputs
        outputs = inputs
        if print_out:
            print(outputs)
        return outputs
    def trainNetwork(self, iterations, learnRate, dataset, **kwargs):
        if 'noOut' in kwargs:
            if kwargs['noOut']:
                print_out = True
        for iteration in range(iterations):
            self.sumError = 0
            self.iterationNumber += 1
            for dataNumber, data in enumerate(dataset):
                inputs = data
                newInputs = []
                for layerNumber, layer in enumerate(self.network[1:]):
                    for neuronNumber, neuron in enumerate(layer):
                        for weightNumber, weight in enumerate(neuron['weights']):
                            neuron['output'] += weight*inputs[weightNumber]
                            neuron['output'] = 1/(1+(math.e**(-float(neuron['output']))))
                            newInputs.append(neuron['output'])
                    inputs = newInputs
                outputs = inputs
                expected = data[-1]
                for expectedNumber, expectedValue in enumerate(expected):
                    self.sumError += (expectedValue-outputs[expectedNumber])**2
                for layerNummber, layer in reversed(enumerate(self.network)):
                    if layerNumber != len(self.network)-1:
                        for neuronNumber, neuron in enumerate(layer):
                            for nextlayerNeuronNumber, nextLayerNeuron in enumerate(self.network[layerNumber]):
                                neuron['error'] += nextLayerNeuron['weights'][neuronNumber]*nextLayerNeuron['error']
                            neuron['error'] *= neuron['output']*(1-neuron['output'])
                    else:
                        for neuronNumber, neuron in layer:
                            neuron['error'] = expected[neuronNumber]-neuron['output']*output*(1-output)
                for layerNumber, layer in enumerate(self.network):
                    inputs = data
                    if layerNumber == 0:
                        inputs = data
                    else:
                        for neuron in self.network[layerNumber-1]:
                            inputs.append(neuron['output'])
                    for neuronNumber, neuron in enumerate(layer):
                        for inputNumber, inputValue in enumerate(inputs[:-2]):
                            neuron['weights'][inputNumber] += (neuron['error']*float(learnRate))*float(inputValue)
                        neuron['weights'][-1] += neuron['error']*learnRate
            if print_out:
                print(f'Iteration: {self.iterationNumber}\t Error: {self.sumError}')
    def getError(self):
        return self.sumError