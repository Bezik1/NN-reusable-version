from helpers.Visualizer import Visualizer
from helpers.Hyperparameters import Hyperparameters
import matplotlib.pyplot as plt
import numpy as np
from const.paths import HYPERPARAMETERS_PATH
from const.visualizer import VISUALIZER_COLUMNS, VISUALIZER_SIZE
from structure.NeuralNetwork import NeuralNetwork
from data.training_data import data_2_prototype, all_y_trues_2S_prototype

if __name__ == "__main__":
    hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)
 
    network = NeuralNetwork(hyperparameters, True)
    loss_history, predicted_values = network.train(data_2_prototype, all_y_trues_2S_prototype, 10)

    visualizer = Visualizer(VISUALIZER_SIZE, VISUALIZER_COLUMNS)
    visualizer.draw(
        [((predicted_values.reshape(data_2_prototype.shape[0])), "Predictions"), (all_y_trues_2S_prototype, "Actual Value")],
        "X",
        "Y",
        "Predictions vs Actual"
    )

    visualizer.draw(
        [(loss_history, "Cost Function")],
        "Epoch",
        "Loss",
        "Cost Function"
    )
    visualizer.visualize()