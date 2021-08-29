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
                neuron['weights'] = [rand() for x in range(structure[layerNumber-1])]
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
        output = inputs
        if print_out:
            print(output)
        return output
                    