import numpy as np

def mse_loss(y, pred):
    return (y - pred)**2

def d_mse_loss(y, pred):
    return -2*(y-pred)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -100, 101)))

def d_sigmoid(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def ReLu(x):
    return np.maximum(0, x)

def d_ReLu(x):
    return 0 if x <= 0 else 1
    