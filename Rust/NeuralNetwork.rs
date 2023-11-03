struct Neuron {
    weights: Vec<f32>,
    output: f32,
    error: f32
}
struct Network {
    layers: Vec<Vec<Neuron>>,
    sum_error: f32,
    trained: bool,
    iteration_number: u64,
    structure: Vec<u64>
}
fn initialize_network(structure: Vec<u64>) -> Network {
    let mut layer: Vec<Neuron>;
    let mut neuron: Neuron;
    let mut fin_layers: Vec<Vec<Neuron>> = Vec::new();
    for layer_num in 1..structure.len() {
        layer = Vec::new();
        for _neuron_num in 0..structure[layer_num] {
            neuron = Neuron{weights: Vec::new(), output: 0.0, error: 0.0};
            if layer_num != structure.len()-1 {
                for _i in 0..structure[layer_num-1]+1 {
                    neuron.weights.push(0.5);
                }
            } else {
                for _i in 0..structure[layer_num-1] {
                    neuron.weights.push(0.5);
                }
            }
            neuron.error = 0.0;
            neuron.output = 0.0;
            layer.push(neuron);
        }
        fin_layers.push(layer);
    }
    let network = Network {
        layers: fin_layers,
        iteration_number: 0,
        sum_error: 0.0,
        trained: false,
        structure: structure
    };
    return network;
}
impl Network {
    fn print_network(&self) {
        for layer_num in 0..self.layers.len() {
            println!("Layer {layer_num}");
            for neuron_num in 0..self.layers[layer_num].len() {
                println!("\t Neuron {neuron_num}");
                println!("\t \t Weights: {:?}", self.layers[layer_num][neuron_num].weights);
                println!("\t \t Output: {}", self.layers[layer_num][neuron_num].output);
                println!("\t \t Error: {}", self.layers[layer_num][neuron_num].error);
            }
        }
    }
    fn activation(&mut self, x:f32, layer_num: usize, neuron_num: usize) -> f32 {
        let x: f32 = x as f32 /self.layers[layer_num][neuron_num].weights.len() as f32;
        //let mut result = (x+x.abs())/2.0;
        let result = 1.0/(1.0+std::f32::consts::E.powf(-x));
        return result;
    }
    fn predict(&mut self, data: &Vec<f32>) -> Vec<f32> {
        let mut inputs: Vec<f32> = Vec::new();
        inputs = data.to_vec();
        let mut new_inputs = Vec::new();
        for layer_num in 0..self.layers.len() {
            for neuron_num in 0..self.layers[layer_num].len() {
                self.layers[layer_num][neuron_num].output = self.layers[layer_num][neuron_num].weights[self.layers[layer_num][neuron_num].weights.len()-1];
                for weight_num in 0..self.layers[layer_num][neuron_num].weights.len() - 1 {
                    self.layers[layer_num][neuron_num].output += self.layers[layer_num][neuron_num].weights[weight_num] * inputs[weight_num];
                }
                self.layers[layer_num][neuron_num].output = self.activation(
                    self.layers[layer_num][neuron_num].output, layer_num, neuron_num
                );
                new_inputs.push(self.layers[layer_num][neuron_num].output);
            }
            inputs = new_inputs;
            new_inputs = Vec::new();
        }
        return inputs;
    }
    fn train_network(&mut self, iterations: i64, learn_rate: f32, input_data: Vec<Vec<f32>>, expected: Vec<Vec<f32>>) {
        for iteration_num in 0..iterations {
            self.sum_error = 0.0;
            self.iteration_number += 1;
            for data_num in 0..input_data.len() {
                let outputs = self.predict(&input_data[data_num]);
                for expected_num in 0..expected[data_num].len() {
                    self.sum_error += (expected[data_num][expected_num]-outputs[expected_num]).powf(2.0);
                }

                // Error backward propagation

                for layer_num in (0..self.layers.len()).rev() {
                    if layer_num != self.layers.len()-1 {
                        for neuron_num in 0..self.layers[layer_num].len() {
                            for neuron_num2 in 0..self.layers[layer_num+1].len() {
                                self.layers[layer_num][neuron_num].error += self.layers[layer_num+1][neuron_num2].weights[neuron_num] * self.layers[layer_num+1][neuron_num2].error;
                            }
                            self.layers[layer_num][neuron_num].error *= self.layers[layer_num][neuron_num].output* (1.0-self.layers[layer_num][neuron_num].output);
                        }
                    } else {
                        for neuron_num in 0..self.layers[layer_num].len() {
                            self.layers[layer_num][neuron_num].error = (expected[data_num][neuron_num]-self.layers[layer_num][neuron_num].output)*(self.layers[layer_num][neuron_num].output)*(1.0-self.layers[layer_num][neuron_num].output);
                        }
                    }
                }

                // weight updating

                for layer_num in 0..self.layers.len() {
                    let mut w_inputs: Vec<f32> = Vec::new();
                    if layer_num != 0 {
                        for neuron_num in 0..self.layers[layer_num-1].len() {
                            w_inputs.push(self.layers[layer_num-1][neuron_num].output);
                        }
                    } else {
                        w_inputs = (&input_data[data_num]).to_vec();
                    }
                    for neuron_num in 0..self.layers[layer_num].len() {
                        for input_num in 0..w_inputs.len()-1 {
                            self.layers[layer_num][neuron_num].weights[input_num] += self.layers[layer_num][neuron_num].error*learn_rate*w_inputs[input_num];
                        }
                        let final_weight_index = self.layers[layer_num][neuron_num].weights.len()-1;
                        self.layers[layer_num][neuron_num].weights[final_weight_index] += self.layers[layer_num][neuron_num].error * learn_rate;
                    }
                }
            }
            println!("Iteration: {iteration_num}, Error: {}", self.sum_error);
            self.trained = true;
        }
    }
}
fn main() {
    let mut structure: Vec<u64> = Vec::new();
    structure.push(2);
    structure.push(4);
    structure.push(4);
    structure.push(2);
    let data_in: Vec<Vec<f32>> = vec![
        vec![0.0,0.0],
        vec![1.0,0.0],
        vec![0.0,1.0],
        vec![1.0,1.0]
    ];
    let data_out: Vec<Vec<f32>> = vec![
        vec![1.0,0.0],
        vec![1.0,0.0],
        vec![1.0,0.0],
        vec![0.0,1.0]
    ];
    let mut network = initialize_network(structure);
    network.print_network();
    network.train_network(100000, 0.3, data_in, data_out);
    network.print_network();
}
