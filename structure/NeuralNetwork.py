import json
import numpy as np
from structure.Layer import Layer
from helpers.functions import mse_loss, d_mse_loss
from const.paths import TRAINED_NETWORK

class NeuralNetwork:
    def __init__(self, hyperparameters, show_training) -> None:
        # Hyperparameters
        self.input_size = hyperparameters.input_size
        self.hidden_size = hyperparameters.hidden_size
        self.hidden_neurons = hyperparameters.hidden_neurons
        self.output_size = hyperparameters.output_size
        self.epochs = hyperparameters.epochs
        self.learning_rate = hyperparameters.learning_rate
        self.regularization_strength = hyperparameters.regularization_strength

        # Dev Tools
        self.show_training = show_training

        # Layers
        self.input_layer = Layer(self.input_size, self.hidden_neurons)
        self.hidden_layers = [Layer(self.hidden_neurons, self.hidden_neurons) for _ in range(self.hidden_size)]
        self.output_layer = Layer(self.hidden_neurons, self.output_size)

    def calculate_regularization_loss(self):
        reg_loss = 0
        for layer in [self.input_layer] + self.hidden_layers + [self.output_layer]:
            for neuron in layer.neurons:
                reg_loss += 0.5 * self.regularization_strength * np.sum(neuron.w**2)
        return reg_loss

    def forward(self, X):
        input_output = self.input_layer.forward(X)
 
        temp = input_output
        for layer in self.hidden_layers:
            temp = layer.forward(temp)

        output = self.output_layer.forward(temp)
        return output

    def backprop(self, y, output):
        d_L_d_out = d_mse_loss(y, output)

        d_L_d_prev = self.output_layer.backprop(d_L_d_out, self.learning_rate)
        for i, layer in enumerate(reversed(self.hidden_layers)):
            d_L_d_prev = layer.backprop(d_L_d_prev, self.learning_rate)
        
        self.input_layer.backprop(d_L_d_prev, self.learning_rate)

    def train(self, X, Y, batch_size):
        loss_history = []
        predictions = []
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_predictions = []
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]

                batch_prediction = []
                for x, y in zip(X_batch, Y_batch):
                    output = self.forward(x)
                    batch_prediction.append(output)

                    loss = mse_loss(y, output) + self.calculate_regularization_loss()
                    epoch_loss += loss
                    self.backprop(y, output)
                epoch_predictions.append(batch_prediction)

            average_loss = epoch_loss / len(X)
            loss_history.append(average_loss)
            predictions.append(epoch_predictions)
            if self.show_training:
                print(f'Epoch: {epoch}; Average Loss: {average_loss[0]}')
                
        self.save_weights(TRAINED_NETWORK)
        return loss_history, np.array(predictions[-1])

    def save_weights(self, filename):
        weights_data = {
            "input_layer": self.input_layer.get_weights(),
            "hidden_layers": [layer.get_weights() for layer in self.hidden_layers],
            "output_layer": self.output_layer.get_weights()
        }

        json_file = open(filename, "w")
        json.dump(weights_data, json_file, indent=2)

    def load_weights(self, filename):
        json_file = open(filename, "r")
        weights_data = json.load(json_file)

        self.input_layer.set_weights(weights_data["input_layer"])
        for i, layer in enumerate(self.hidden_layers):
            layer.set_weights(weights_data["hidden_layers"][i])
        self.output_layer.set_weights(weights_data["output_layer"])