import numpy as np

def mean_subtraction(x_data):
    return x_data - np.mean(x_data)

def normalization(x_data):
    return x_data / np.std(x_data, axis=0)

