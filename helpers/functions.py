import numpy as np

def mse_loss(y, pred):
    return (y - pred)**2

def d_mse_loss(y, pred):
    return -2*(y-pred)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def d_sigmoid(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))