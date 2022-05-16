class NeuralNetwork {
    constructor(structure, random=false){
        this.sumError = 0.0;
        this.network = []
        this.trained = false
        this.iterationNumber = 0
        this.structure = structure
        for (let layerNumber = 1; layerNumber < structure.length; layerNumber++){
            let layer = structure[layerNumber]
            let newLayer = []
            for (let neuronNumber = 0; neuronNumber < layer; neuronNumber++){
                let neuron = {}
                if (layerNumber != structure.length-1){
                    if (!random){
                        neuron['weights'] = []
                        for (let x = 0; x < structure[layerNumber-1]+1; x++){
                            neuron['weights'].push(0.5)
                        }
                    }
                    else{
                        neuron['weights'] = []
                        for (let x = 0; x < structure[layerNumber-1]+1; x++){
                            neuron['weights'].push(Math.random())
                        }
                        
                    }
                }
                else {
                    if (!random){
                        neuron['weights'] = [] //0.5 for x in range(structure[layerNumber-1])
                        for (let x = 0; x < structure[layerNumber-1]; x++){
                            neuron['weights'].push(0.5)
                        }
                    }
                    else{
                        neuron['weights'] = [] //0.5 for x in range(structure[layerNumber-1])
                        for (let x = 0; x < structure[layerNumber-1]; x++){
                            neuron['weights'].push(Math.random())
                        }
                    }
                }
                neuron['error'] = 0.0
                neuron['output'] = 0.0
            newLayer.push(neuron)
            }
        this.network.push(newLayer)
        }
    }
    // def importNetwork(self, directory, **kwargs):
    //     json.load(open(f"{directory}", 'r'))

    // def exportNetwork(self, directory, **kwargs):
    //     with open(f'{directory}', 'w') as file:
    //         json.dump(self.network, file)

    predict(data){
        let inputs = data
        // Starting the forward propagation section
        for (let layerNumber = 0; layerNumber < this.network.length; layerNumber++){
            let layer = this.network[layerNumber]
            let newInputs = []
            for (let neuronNumber = 0; neuronNumber < layer.length; neuronNumber++){
                this.network[layerNumber][neuronNumber]['output'] = this.network[layerNumber][neuronNumber]['weights'][this.network[layerNumber][neuronNumber]['weights'].length-1]
                for (let weightNumber = 0; weightNumber < this.network[layerNumber][neuronNumber]['weights'].slice(0,-1).length; weightNumber++){
                    this.network[layerNumber][neuronNumber]['output'] += this.network[layerNumber][neuronNumber]['weights'][weightNumber]*inputs[weightNumber]
                }
                this.network[layerNumber][neuronNumber]['output'] = 1 / (1+(Math.E**(-this.network[layerNumber][neuronNumber]['output'])))
                newInputs.push( this.network[layerNumber][neuronNumber]['output'])
            }
            inputs = newInputs
            newInputs = []
        }
        return inputs
    }

    trainNetwork(iterations, learnRate, dataset, out=True) {
        for (let iteration = 0; iteration < iterations; iteration++){
            this.sumError = 0
            this.iterationNumber += 1
            for (let dataNumber = 0; dataNumber < dataset.length; dataNumber++) {
                let data = dataset[dataNumber]
                let inputs = data.slice(0,-1)
                // Starting the forward propagation section
                for (let layerNumber = 0; layerNumber < this.network.length; layerNumber++){
                    let layer = this.network[layerNumber]
                    let newInputs = []
                    for (let neuronNumber = 0; neuronNumber < layer.length; neuronNumber++){
                        this.network[layerNumber][neuronNumber]['output'] = this.network[layerNumber][neuronNumber]['weights'][this.network[layerNumber][neuronNumber]['weights'].length-1]
                        for (let weightNumber = 0; weightNumber < this.network[layerNumber][neuronNumber]['weights'].slice(0,-1).length; weightNumber++){
                            this.network[layerNumber][neuronNumber]['output'] += this.network[layerNumber][neuronNumber]['weights'][weightNumber]*inputs[weightNumber]
                        }
                        this.network[layerNumber][neuronNumber]['output'] = 1 / (1+(Math.E**(-this.network[layerNumber][neuronNumber]['output'])))
                        newInputs.push( this.network[layerNumber][neuronNumber]['output'])
                    }
                    inputs = newInputs
                    newInputs = []
                }
                let outputs = inputs
                let expected = data[data.length-1]
                for (let expectedNumber = 0; expectedNumber < expected.length; expectedNumber++){
                    let expectedValue = expected[expectedNumber]
                    this.sumError += (expectedValue-outputs[expectedNumber])**2
                }
                    
                // Error backward propagation
                for (let layerNumber = this.network.length-1; layerNumber >= 0; layerNumber--){
                    if (layerNumber != this.network.length-1){
                        for (let neuronNumber = 0; neuronNumber < this.network[layerNumber].length; neuronNumber++){
                            for (let neuronNumber2 = 0; neuronNumber2 < this.network[layerNumber+1].length; neuronNumber2++){
                                this.network[layerNumber][neuronNumber]['error'] += this.network[layerNumber+1][neuronNumber2]['weights'][neuronNumber]*this.network[layerNumber+1][neuronNumber2]['error']
                            }
                            this.network[layerNumber][neuronNumber]['error'] *= this.network[layerNumber][neuronNumber]['output']*(1-this.network[layerNumber][neuronNumber]['output'])
                        }
                    }
                    else {
                        for (let neuronNumber = 0; neuronNumber < this.network[layerNumber].length; neuronNumber++){
                            this.network[layerNumber][neuronNumber]['error'] = (expected[neuronNumber]-this.network[layerNumber][neuronNumber]['output'])*(this.network[layerNumber][neuronNumber]['output']*(1-this.network[layerNumber][neuronNumber]['output']))
                        }
                    }
                }
                // Updates the weights
                for (let layerNumber = 0; layerNumber < this.network.length; layerNumber++){
                    let inputs = []
                    if (layerNumber != 0){
                        for (let neuronNumber = 0; neuronNumber < this.network[layerNumber-1].length; neuronNumber++){
                            inputs.push(this.network[layerNumber-1][neuronNumber]['output'])
                        }
                    }
                    else {
                        inputs = data
                    }
                    for (let neuronNumber = 0; neuronNumber < this.network[layerNumber].length; neuronNumber ++){
                        for (let inputNumber = 0; inputNumber < inputs.length-1; inputNumber++){
                            this.network[layerNumber][neuronNumber]['weights'][inputNumber] = (this.network[layerNumber][neuronNumber]['weights'][inputNumber]) + ((learnRate)*(this.network[layerNumber][neuronNumber]['error'])*(inputs[inputNumber]))
                        }
                        this.network[layerNumber][neuronNumber]['weights'][this.network[layerNumber][neuronNumber]['weights'].length-1] += this.network[layerNumber][neuronNumber]['error']*learnRate
                    }
                }
            }
            if (out){
                console.log(`Iteration: ${this.iterationNumber}\t Error: ${this.sumError}`)
            }
        }
    }
    getError(){
        return this.sumError
    }
}

let test = ['a','b','c']
let network = new NeuralNetwork([2, 3, 3, 2], true)
let dataset = [
    [0,0,[1,0]],
    [1,1,[0,1]],
    [0,1,[0,1]],
    [1,0,[0,1]]
]


network.trainNetwork(10000, 0.5, dataset, true)
console.log(JSON.stringify(network.network))
// network.trainNetwork(1, 0.5, dataset, true)
// console.log(JSON.stringify(network.network))
console.log(network.predict([1,1]))