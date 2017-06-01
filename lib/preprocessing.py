import numpy as np

def mean_subtraction(x_data):
    return x_data - np.array([np.mean(x_data, axis=1)]).T

def normalization(x_data):
    return x_data / np.std(x_data, axis=0)