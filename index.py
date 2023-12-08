import numpy as np

from const.paths import TRAINED_NETWORK_PARAMETERS
from helpers.gender import get_gender
from helpers.Hyperparameters import Hyperparameters
from const.paths import HYPERPARAMETERS_PATH
from const.offsets import HEIGHT_OFFSET, WEIGHT_OFFSET
from structure.NeuralNetwork import NeuralNetwork
from data.training_data import data

hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)

# Train our neural network!
network = NeuralNetwork(hyperparameters, False)
network.load_trained_network(TRAINED_NETWORK_PARAMETERS)

weight = float(input("Weight: "))
height = float(input("Height: "))

data = np.array([weight - WEIGHT_OFFSET, height - HEIGHT_OFFSET])

probability = network.feedforward(data)
gender, precent = get_gender(probability)
print(f"You're a {gender} for {precent}%")