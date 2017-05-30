from lib.layers.layer import Layer
import gnumpy as gp
import numpy as np


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):
        self.input = inp

        # if inp > 0:
        #     return inp
        # return 0.01 * inp

        tmp_grt = gp.garray(inp) > gp.zeros(inp.shape)
        return gp.garray(inp) * tmp_grt

        # Logistic implementation for now :(

        # return gp.log(gp.ones(inp.shape) + gp.exp(inp))

    def get_derivative(self):
        # if self.input > 0:
        #     return 1
        # return 0.01
        # return gp.logistic(self.input)
        return gp.garray(self.input) > gp.zeros(self.input.shape)

    def get_input_gradient(self, output_gradient):
        return output_gradient * self.get_derivative()

