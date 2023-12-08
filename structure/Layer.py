import numpy as np
from structure.Neuron import Neuron
from helpers.functions import d_sigmoid

class Layer:
    def __init__(self, prev_size, size) -> None:
        self.size = size

        self.neurons = [Neuron(prev_size) for _ in range(size)]
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X.flatten()

        self.output = np.array([n.forward(self.input) for n in self.neurons])
        return self.output.flatten()
    
    def backprop(self, d_L_d_out, learning_rate):
            d_out_d_input = np.array([d_sigmoid(neuron.z) for neuron in self.neurons])

            d_L_d_input = d_L_d_out * d_out_d_input

            d_L_d_w = np.zeros_like(self.neurons[0].w)

            for i, neuron in enumerate(self.neurons):
                d_L_d_w += d_L_d_input[i] * neuron.w.flatten()
                neuron.backprop(d_L_d_input[i], learning_rate)

            return d_L_d_w