import json
from random import random as rand
class NeuralNetwork:
    trained = False
    iterationNumber = 0
    errorSave = float()
    SumError = float()
    def __init__(self, structure):
        if not isinstance(structure, list):
            raise TypeError('Structure needs to be a list!')
        NeuralNetwork.network = list()
        for layerNumber in range(1,len(structure)):
            newLayer = []
            for neuronNumber in range(structure[layerNumber]):
                weights = []
                for weightNumber in range(structure[layerNumber-1]+1):
                    weights.append(0.5)
                neuron = {}
                neuron['weights'] = weights
                neuron['error'] = float()
                neuron['output'] = float()
                newLayer.append(neuron)
            NeuralNetwork.network.append(newLayer)
    def trainNetwork(self, iterations, learnRate, dataset, noOut = True):
        if isinstance(iterations, int) == False or iterations < 1:
            raise ValueError('The number of iterations must be a positive integer!')
        NeuralNetwork.trained = True
        def activateNeuron(weights, inputs):
            output = weights[-1]            
            for weightNumber in range(len(weights)-1):
                output += float(weights[weightNumber])*float(inputs[weightNumber])
            return output
        def transfer(activationValue):
            return 1/(1+(2.7182818284590452353602874713527**(-float(activationValue))))
        def forwardPropagation(data):
            inputs = data
            for layer in NeuralNetwork.network:
                newInputs = []
                for neuron in layer:
                    activation = activateNeuron(neuron['weights'], inputs)
                    neuron['output'] = transfer(activation)
                    newInputs.append(neuron['output'])
                print(newInputs)
                inputs = newInputs
            return inputs
        def transferDerivative(output):
            return output*(1-output)
        def errorBackwardPropagation(expected):
            for layerNumber in reversed(range(len(NeuralNetwork.network))):
                layer = NeuralNetwork.network[layerNumber]
                if layerNumber != len(NeuralNetwork.network)-1:
                    for neuronNumber in range(len(layer)):
                        for neuron in NeuralNetwork.network[layerNumber+1]:
                            NeuralNetwork.network[layerNumber][neuronNumber]['error'] += neuron['weights'][neuronNumber]*neuron['error']
                        NeuralNetwork.network[layerNumber][neuronNumber]['error'] = NeuralNetwork.network[layerNumber][neuronNumber]['error']*transferDerivative(NeuralNetwork.network[layerNumber][neuronNumber]['output'])
                else:
                    for neuronNumber in range(len(layer)):
                        neuron = layer[neuronNumber]
                        neuron['error'] = (expected[neuronNumber]-neuron['output'])*transferDerivative(neuron['output'])
        def updateWeights(data, learnRate):
            for layerNumber in range(len(NeuralNetwork.network)):
                inputs = []
                if layerNumber != 0:
                    for neuron in NeuralNetwork.network[layerNumber-1]:
                        inputs.append(neuron['output'])
                else:
                    inputs = data
                layer = NeuralNetwork.network[layerNumber]
                for neuronNumber in range(len(layer)):
                    neuron = layer[neuronNumber]
                    for inputNumber in range(len(inputs)-1):
                        neuron['weights'][inputNumber] = float(neuron['weights'][inputNumber]) + (float(learnRate)*float(neuron['error'])*float(inputs[inputNumber]))
                    neuron['weights'][-1] += neuron['error']*learnRate
        for iteration in range(iterations):
            NeuralNetwork.SumError = 0
            NeuralNetwork.iterationNumber += 1
            for data in dataset:
                outputs = forwardPropagation(data)
                expected = data[-1]
                for expectedNumber in range(len(expected)):
                    NeuralNetwork.SumError += (expected[expectedNumber]-outputs[expectedNumber])**2
                errorBackwardPropagation(expected)
                updateWeights(data, learnRate)
            if noOut == False:
                continue
            if noOut == True:
                print('Iteration: '+str(NeuralNetwork.iterationNumber)+', Error: '+str(NeuralNetwork.SumError))
        NeuralNetwork.trained = True
    def predict(self, data):
        def activateNeuron(weights, inputs):
            output = weights[-1]
            for weightNumber in range(len(weights)-1):
                output += weights[weightNumber]*inputs[weightNumber]
            return output
        def transfer(activationValue):
            return 1/(1+(2.7182818284590452353602874713527**(-float(activationValue))))
        def forwardPropagation(data):
            inputs = data
            for layer in NeuralNetwork.network:
                newInputs = []
                for neuron in layer:
                    activation = activateNeuron(neuron['weights'], inputs)
                    neuron['output'] = transfer(activation)
                    newInputs.append(neuron['output'])
                inputs = newInputs
            return inputs
        outputs = forwardPropagation(data)
        return outputs
    def exportNetwork(self, directory):
        with open(directory, 'w') as networkFile:
            json.dump(self.network, networkFile)
    def importNetwork(self, directory):
        NeuralNetwork.network = json.load(open(directory, 'r'))
    def getError(self):
        return self.SumError

if __name__ == "__main__":
    network = NeuralNetwork([2,3,2])
    dataset = [
        [0,0,(1,0)],
        [0,1,(1,0)],
        [1,0,(1,0)],
        [1,1,(0,1)]
    ]
    network.trainNetwork(1000, 0.6, dataset)
    print(network.predict([0,0]))
    print(network.network)
