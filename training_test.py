import numpy as np
from helpers.gender import get_gender
from helpers.Visualizer import Visualizer
from helpers.Hyperparameters import Hyperparameters
from const.paths import HYPERPARAMETERS_PATH
from const.offsets import HEIGHT_OFFSET, WEIGHT_OFFSET
from const.visualizer import VISUALIZER_COLUMNS, VISUALIZER_SIZE
from structure.NeuralNetwork import NeuralNetwork
from data.training_data import data, all_y_trues

if __name__ == "__main__":
    hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)
    
    network = NeuralNetwork(hyperparameters, True)
    loss_history, last_predictions = network.train(data, all_y_trues)

    visualizer = Visualizer(VISUALIZER_SIZE, VISUALIZER_COLUMNS)
    visualizer.draw(
        [(loss_history, "Cost Function")],
        "Epoch",
        "Loss",
        "Cost Function"
    )

    visualizer.draw(
        [(last_predictions, "Predictions"), (all_y_trues, "Actual Value")],
        "Sample",
        "Gender",
        "Predictions vs Actual"
    )

    visualizer.visualize()

    weight = float(input("Weight: "))
    height = float(input("Height: "))

    data = np.array([weight - WEIGHT_OFFSET, height - HEIGHT_OFFSET])

    probability = network.forward(data)
    gender, precent = get_gender(probability[0])
    print(f"You're a {gender} for {precent}%")