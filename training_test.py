from helpers.Visualizer import Visualizer
from helpers.Hyperparameters import Hyperparameters
from const.paths import HYPERPARAMETERS_PATH
from structure.NeuralNetwork import NeuralNetwork
from data.training_data import data_2, all_y_trues_2S

if __name__ == "__main__":
    hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)
 
    network = NeuralNetwork(hyperparameters, True)
    loss_history, last_predictions = network.train(data_2, all_y_trues_2S)

    visualizer = Visualizer(1, 1)

    visualizer.draw(
        [(last_predictions, "Predictions"), (all_y_trues_2S, "Actual Value")],
        "X",
        "Y",
        "Predictions vs Actual"
    )

    visualizer.visualize()