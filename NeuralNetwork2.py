import json
from random import random as rand
class NeuralNetwork:
    def __init__(self,structure):
        self.sumError = float()
        self.network = []
        self.trained = False
        self.iterationNumber = 0
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