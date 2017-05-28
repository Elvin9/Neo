import gnumpy as gp
import numpy as np
from lib.loss_functions.loss import Loss


class MSELoss(Loss):

    def __init__(self):
        super().__init__()

    def get_output(self, inp, expected):

        self.input = inp

        error = self.input - expected
        self.output = (1.0 / inp.shape[0]) * 0.5 * gp.dot(error.T, error)
        return np.diag(self.output.as_numpy_array())

    def get_input_gradient(self, prediction, inp=None):

        if not inp:
            return prediction - self.input
        else:
            return prediction - inp
