import numpy as np
from helpers.gender import get_gender
from helpers.Visualizer import Visualizer
from helpers.Hyperparameters import Hyperparameters
from const.paths import HYPERPARAMETERS_PATH, TRAINED_NETWORK
from structure.NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)
 
    network = NeuralNetwork(hyperparameters, True)
    network.load_weights(TRAINED_NETWORK)
    
    print(network.forward(np.array([6]))*100)

    # weight = float(input("Weight: "))
    # height = float(input("Height: "))

    # data = np.array([weight - WEIGHT_OFFSET, height - HEIGHT_OFFSET])

    # probability = network.forward(data)
    # gender, precent = get_gender(probability[0])
    # print(f"You're a {gender} for {precent}%")