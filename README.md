# Neural Network Module

To import the neural network class, add the following code.

```from NeuralNetwork import NeuralNetwork```

The rest of these instructions will be based on the `network` variable created in the following line.

```network = NeuralNetwork([2,3,2])```

The list that is given as a parameter to the the class constructor is the structure of the neural network. This example means that the network will have 2 input nodes, 3 hidden nodes, and then 2 output nodes. By adding more nodes in the middle of the list, you can make more hidden layers (i.e. `[2,3,3,2]` would make a network with an additional hidden layer with 3 neurons). 

This is the sample dataset that will be used throughout the rest of these instructions in the `dataset` variable.

```
dataset = [
    [0,0,(1,0)],
    [0,1,(1,0)],
    [1,0,(1,0)],
    [1,1,(0,1)]
]
```

This dataset will be used to train the neural network in the example. This is how you train the network:

`network.trainNetwork(1000, 0.6, dataset, False)`

This code will train the network with 1000 iterations and a learn rate of 0.6. The dataset is then passed to allow the network to be trained. The `False` being pasesed as the final parameter is optional and set to `True` by default. When `True`, this function will print out information about the networks progress in learning with the data. You can test the network using the `predict` method. It can be called like this:

`output = network.predict([1,1])`

For this data, you need to pass the data without the solution and the network will attempt to guess what the answer should be based on the data that it has been trained with. After training, you may need to save the network to a file. This can be done using the following code:

`network.exportNetwork('network.neural')`

This will create a new file if it is not already present and will save the neural network in a JSON format. A compressed option may become available in a later version. To load a network back into a network variable in another program or when starting the program again, you use the following code:

`network.importNetwork('network.neural')`

For logging and statistics, the `getError()` function is included and is used like this:

`network.getError()`

### Features coming soon:

- Q-Learning System
- Easy Adversarial Networks
- More multithreading optimization
- Javascript version of RAMSaverNetwork
- More documentation coming soon at http://docs.terminotech.com/Neural-Network-Module (does not work yet)


If you have any questions or suggestions and don't want to go through GitHub, you can reach me at willetsjared@gmail.com.