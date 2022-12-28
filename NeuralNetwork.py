import json
from math import e as euler
from random import random as rand
import threading
import time

class NeuralNetwork:
	def __init__(self, structure, random=False):
		self.sumError = float()
		self.network = []
		self.trained = False
		self.iterationNumber = 0
		self.structure = structure
		for layerNumber, layer in enumerate(structure[1:]):
			newLayer = []
			for neuronNumber in range(layer):
				neuron = dict()
				if layerNumber != len(structure)-1:
					if not random:
						neuron['weights'] = [0.5 for x in range(structure[layerNumber-1]+1)]
					else:
						neuron['weights'] = [rand() for x in range(structure[layerNumber-1]+1)]
				else:
					if not random:
						neuron['weights'] = [0.5 for x in range(structure[layerNumber-1])]
					else:
						neuron['weights'] = [rand() for x in range(structure[layerNumber-1])]
				neuron['error'] = float()
				neuron['output'] = float()
				newLayer.append(neuron)
			self.network.append(newLayer)

	def importNetwork(self, directory, **kwargs):
		json.load(open(directory, 'r'))

	def exportNetwork(self, directory, **kwargs):
		with open(directory, 'w') as file:
			json.dump(self.network, file)

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

	def trainNetwork(self, iterations: int, learnRate: float, dataset:list, out=True):
		# previousError = 50000000
		print_out = out ## Sets whether or not the iteration statement will be printed
		for iteration in range(iterations):
			self.sumError = 0
			self.iterationNumber += 1
			for dataNumber, data in enumerate(dataset):
				inputs = data[:-1]
				for inputValue in inputs:
					inputValue = float(inputValue) 
				## Starting the forward propagation section
				for layerNumber, layer in enumerate(self.network):
					newInputs = []
					for neuronNumber, neuron in enumerate(layer):
						neuron['output'] = neuron['weights'][-1]
						for weightNumber, weight in enumerate(neuron['weights'][:-1]):
							neuron['output'] += weight*inputs[weightNumber]
						neuron['output'] = 1 / (1+(euler**(-float(neuron['output']))))
						newInputs.append(neuron['output'])
					inputs = newInputs
					newInputs = []
				outputs = inputs
				expected = data[-1]
				for expectedNumber, expectedValue in enumerate(expected):
					self.sumError += (expectedValue-outputs[expectedNumber])**2
				# Error backward propagation
				for layerNumber in reversed(range(len(self.network))):
					layer = self.network[layerNumber]
					if layerNumber != len(self.network)-1:
						for neuronNumber in range(len(layer)):
							for neuron in self.network[layerNumber+1]:
								self.network[layerNumber][neuronNumber]['error'] += neuron['weights'][neuronNumber]*neuron['error']
							self.network[layerNumber][neuronNumber]['error'] = self.network[layerNumber][neuronNumber]['error']*(
								self.network[layerNumber][neuronNumber]['output']*(1-self.network[layerNumber][neuronNumber]['output']))
					else:
						for neuronNumber in range(len(layer)):
							neuron = layer[neuronNumber]
							neuron['error'] = (
								expected[neuronNumber]-neuron['output'])*(neuron['output']*(1-neuron['output']))
				# Updates the weights
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
				print('Iteration: '+str(self.iterationNumber)+'\t Error: '+str(self.sumError))

	def getError(self):
		return self.sumError

class RAMSaverNeuralNetwork:
	def __init__(self, structure, name:str, random=False):
		import os
		self.sumError = float()
		self.network = []
		self.trained = False
		self.iterationNumber = 0
		self.structure = structure
		self.currentlySaving = []
		self.stage1Iteration = 0
		self.stage2Iteration = 0
		self.stage3Iteration = 0
		self.laodedArray = []
		self.asynchronousBackwardPropagation = False
		try:os.mkdir(name)
		except:pass
		logging.basicConfig(filename='test.log', level=logging.DEBUG)
		for layerNumber, layer in enumerate(structure[1:]):
			try:os.mkdir(name+"/Layer"+str(layerNumber))
			except:pass
			layerNumber += 1
			newLayer = []
			for neuronNumber in range(layer):
				neuron = dict()
				filename = name+"/Layer"+str(layerNumber-1)+"/"+str(neuronNumber)+".neuron"
				file = open(filename,'w')
				if layerNumber != len(structure)-1:
					if not random:
						neuron['weights'] = [0.5 for x in range(structure[layerNumber-1]+1)]
					else:
						neuron['weights'] = [rand() for x in range(structure[layerNumber-1]+1)]
				else:
					if not random:
						neuron['weights'] = [0.5 for x in range(structure[layerNumber-1])]
					else:
						neuron['weights'] = [rand() for x in range(structure[layerNumber-1])]
				neuron['error'] = float()
				neuron['output'] = float()
				json.dump(neuron, file)
				file.close()
				newLayer.append(name+"/Layer"+str(layerNumber-1)+"/"+str(neuronNumber)+".neuron")
			self.network.append(newLayer)

	def getNeuron(self,directory):
		with open(directory,'rt+') as file:
			string = file.read()
			neuron = json.loads(string)
			file.close()
		return neuron
		
	def saveNeuron(self, directory, neuron):
		testNeuron = neuron
		with open(directory,'wt') as writeFile:
			writeFile.write(json.dumps(testNeuron))
			writeFile.close()

	def importNetwork(self, name, structure, mode = 'r'):
		for layerNumber, layer in enumerate(structure[1:]):
			newLayer = []
			for neuronNumber in range(layer):
				filename = name+"/Layer"+str(layerNumber)+"/"+str(neuronNumber)+".neuron"
				newLayer.append(filename)
			self.network.append(newLayer)

	def predict(self, data, **kwargs):
		print_out = False
		if 'print_out' in kwargs:
			print_out = True
		inputs = data
		newInputs = []
		for layerNumber, layer in enumerate(self.network[1:]):
			for neuronNumber, neuron in enumerate(layer):
				neuron = self.getNeuron(neuron)
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
	
	def trainNetwork(self, iterations: int, learnRate: float, dataset:list, out=True):
		print_out = out ## Sets whether or not the iteration statement will be printed
		totalStartTime = time.time()
		for iteration in range(iterations):
			startTime = time.time()
			self.sumError = 0
			self.iterationNumber += 1
			for dataNumber, data in enumerate(dataset):
				
				# Starting the forward propagation section (Changing output)
				self.forwardPropagateThread(data)
				# threading.Thread(target = forwardPropagateThread, args = (self,data))
				# Error backward propagation (Changing error)
				self.backwardPropagationThread()
				# threading.Thread(target = backwardPropagationThread, args = (self))
				# Weight updating (Changing weights)
				self.updateWeightsThread(learnRate, data)
				# threading.Thread(target = updateWeghtsThread, args=(self,learnRate, data))
			endTime = time.time()
			if print_out:
				print('Iteration: '+str(self.iterationNumber)+'\t Error: '+str(self.sumError)+"\t Time: "+str(endTime-startTime)+"\t Total Time: "+str(endTime-totalStartTime))

	def getError(self):
		self.loaded = True
		return self.sumError

	def updateWeightsThread(self, learnRate, data):
		# Updates the weights (Changing weights)
		for layerNumber in range(len(self.network)):
			inputs = []
			if layerNumber != 0:
				for neuron in self.network[layerNumber-1]:
					neuron = self.getNeuron(neuron)
					inputs.append(neuron['output'])
			else:
				inputs = data
			layer = self.network[layerNumber]
			for neuronNumber in range(len(layer)):
				neuronDir = layer[neuronNumber]
				neuron1 = self.getNeuron(layer[neuronNumber])
				testNeuron = self.getNeuron(layer[neuronNumber])
				neuron = neuron1
				for inputNumber in range(len(inputs)-1):
					neuron['weights'][inputNumber] = float(neuron['weights'][inputNumber]) + (float(learnRate)*float(neuron['error'])*float(inputs[inputNumber]))
				neuron['weights'][-1] += neuron['error']*learnRate
				if neuron != testNeuron and neuronDir == "newNet/Layer1/0.neuron":
					self.saveNeuron(neuronDir, neuron)
		self.stage3Iteration += 1
	def forwardPropagateThread(self, data):
		inputs = data[:-1]
		for inputValue in inputs:
			inputValue = float(inputValue) 
		for layerNumber, layer in enumerate(self.network):
			newInputs = []
			for neuronNumber, neuron in enumerate(layer):
				neuronDir = neuron
				neuron = self.getNeuron(neuron)
				neuron['output'] = neuron['weights'][-1]
				for weightNumber, weight in enumerate(neuron['weights'][:-1]):
					neuron['output'] += weight*inputs[weightNumber]
				neuron['output'] = 1 / (1+(euler**(-float(neuron['output']))))
				newInputs.append(neuron['output'])
				self.saveNeuron(neuronDir, neuron)
			inputs = newInputs
			newInputs = []
		outputs = inputs
		expected = data[-1]
		for expectedNumber, expectedValue in enumerate(expected):
			self.sumError += (expectedValue-outputs[expectedNumber])**2
		self.expected = expected
		self.stage1Iteration +=1
	
	def backPropagationNeuronThread(self, layerNumber, neuronNumber):
		tempNeuron = self.getNeuron(self.network[layerNumber][neuronNumber])
		tempNeuronDir = self.network[layerNumber][neuronNumber]
		for neuron in self.network[layerNumber+1]:
			neuron = self.getNeuron(neuron)
			tempNeuron['error'] += neuron['weights'][neuronNumber]*neuron['error']
			tempNeuron['error'] *= (tempNeuron['output']*(1-tempNeuron['output']))
		self.saveNeuron(tempNeuronDir, tempNeuron)
		self.neuronQueue.remove(neuronNumber)
		
	def backwardPropagationThread(self):
		for layerNumber in reversed(range(len(self.network))):
			layer = self.network[layerNumber]
			if layerNumber != len(self.network)-1:
				self.neuronQueue = []
				for neuronNumber in range(len(layer)):
					if self.asynchronousBackwardPropagation:
						self.neuronQueue.append(neuronNumber)
						x = threading.Thread(target = self.backPropagationNeuronThread, args = (layerNumber, neuronNumber), daemon=True)
						x.start()
					else:
						self.neuronQueue.append(neuronNumber)
						self.backPropagationNeuronThread(layerNumber, neuronNumber)
				while len(self.neuronQueue) != 0 and self.asynchronousBackwardPropagation:
					continue
			else:
				for neuronNumber in range(len(layer)):
					neuronDir = layer[neuronNumber]
					neuron = self.getNeuron(layer[neuronNumber])
					neuron['error'] = (
						self.expected[neuronNumber]-neuron['output'])*(neuron['output']*(1-neuron['output']))
					self.saveNeuron(neuronDir, neuron)
		self.stage2Iteration += 1
	
