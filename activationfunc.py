import numpy as np


def relu(x):
    return (x > 0) * x

def relu_deriv(x):
    return x > 0
    
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)