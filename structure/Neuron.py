import numpy as np
from helpers.functions import d_tanh, tanh

class Neuron:
    def __init__(self, input_size)-> None:
        self.w = np.random.rand(input_size)
        self.b = 0

        self.input = None
        self.d_l_d_w = None
        self.d_l_d_b = None
        self.z = None
        self.output = None

    def calcualte_z(self, X):
        """Calculating sum of Wi multiplied by Xi and b"""
        self.z = np.dot(self.w, X) + self.b
        return self.z

    def forward(self, X):
        self.input = X
        self.calcualte_z(X)

        self.output = tanh(self.z)
        return self.output
    
    def backprop(self, d_L_d_out, learning_rate):
        d_out_d_z = d_tanh(self.z)

        d_L_d_z = d_L_d_out * d_out_d_z

        for i in range(len(self.w)):
            self.w[i] += learning_rate * d_L_d_z * self.input[i]
        
        self.b += learning_rate * d_L_d_z

        return d_L_d_z * self.w.flatten()
    
    def get_weights(self):
        return {"weights": self.w.tolist(), "bias": self.b.tolist()}

    def set_weights(self, weights_data):
        self.w = np.array(weights_data["weights"])
        self.b = np.array(weights_data["bias"])