import gnumpy as gp
import numpy as np
from lib.loss_functions.loss import Loss


class CrossEntropy(Loss):

    def __init__(self):
        super().__init__()

    # def get_output(self, inp, target):
    #
    #     self.input = inp
    #     errors = (target * gp.log(self.input)) + ((1 - target) * gp.log(1 - self.input))
    #
    #     return -(1.0 / self.input.shape[0]) * gp.sum(errors, axis=0)
    #
    # def get_input_gradient(self, target):
    #     numer = (-2 * target * self.input) + target + self.input
    #     denumer = self.input - (self.input * self.input)
    #     inv_denumer = gp.garray(1 / denumer.as_numpy_array())
    #
    #     return -(1.0 / self.input.shape[0]) * inv_denumer * numer

    def get_output(self, inp, target):
        self.input = inp
        return gp.sum(target * gp.log(self.input), axis=0)

    def get_input_gradient(self, target):
        denumer = gp.garray(1.0 / self.input.as_numpy_array())
        return -(1.0 / self.input.shape[0]) * target * denumer