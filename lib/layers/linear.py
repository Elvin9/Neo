import gnumpy as gp
import numpy as np
from .layer import Layer


class Linear(Layer):

    def __init__(self, input_num, output_num, bias=True):
        super().__init__()

        self.bias = bias

        if self.bias:
            input_num += 1

        self.weights = (gp.rand(output_num, input_num))

    def get_output(self, inp):

        if self.bias:
            self.input = gp.concatenate((inp, gp.ones((1, inp.shape[1]))))
        else:
            self.input = inp

        self.output = gp.dot(self.weights, self.input)
        return self.output

    def get_input_gradient(self, output_gradient):
        gradient = gp.dot(self.weights.T, output_gradient)
        return gradient

    def get_parameter_gradient(self, output_gradient):
        gradient = gp.dot(output_gradient, self.input.T)
        return gradient

    def update_parameters(self, output_gradient, rate):
        self.weights = self.weights - (rate * self.get_parameter_gradient(output_gradient))

