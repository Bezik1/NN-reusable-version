import numpy as np
from helpers.Visualizer import Visualizer
from helpers.Hyperparameters import Hyperparameters
from const.paths import HYPERPARAMETERS_PATH, TRAINED_NETWORK
from structure.NeuralNetwork import NeuralNetwork
from data.training_data import data_2, all_y_trues_2S

if __name__ == "__main__":
    hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)
 
    network = NeuralNetwork(hyperparameters, True)
    network.load_weights(TRAINED_NETWORK)
    predictions = [network.forward(x) for x in data_2]
    
    visualizer = Visualizer(1, 1)
    visualizer.draw(
        [(predictions, "Predictions"), (all_y_trues_2S, "Actual Value")],
        "X",
        "Y",
        "Predictions vs Actual"
    )

    visualizer.visualize()

    while True:
        number = float(input("Number to square: "))
        print(round(network.forward(np.array([number / 100]))[0]*10000))