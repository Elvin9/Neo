import gnumpy as gp
import numpy as np
from lib.layers.layer import Layer


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):
        # normalize the input to be numerically stable
        inp = inp - gp.max(inp.T, axis=1)

        exp_sum = gp.sum(gp.exp(inp), axis=0).as_numpy_array()
        inv_exp_sum = gp.garray((1 / exp_sum))
        inv_exp_sum = gp.tile(inv_exp_sum, (inp.shape[0], 1))

        self.output = gp.exp(inp) * inv_exp_sum
        return self.output

    def get_derivative(self):
        sub_prod = 1 - self.output
        return self.output * sub_prod

    def get_input_gradient(self, output_gradient):
        return output_gradient * self.get_derivative()


class SoftmaxCrossEntropyLayer(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):

        # normalize the input to be numerically stable
        inp = inp - gp.max(inp.T, axis=1)

        exp_sum = gp.sum(gp.exp(inp), axis=0).as_numpy_array()
        inv_exp_sum = gp.garray((1 / exp_sum))
        inv_exp_sum = gp.tile(inv_exp_sum, (inp.shape[0], 1))

        self.output = gp.exp(inp) * inv_exp_sum
        return self.output

    def get_input_gradient(self, output_gradient):
        # the SoftmaxCrossEntropyLoss did that for both of us
        return output_gradient

