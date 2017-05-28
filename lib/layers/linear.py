import theano.tensor as T
import numpy as np
from .layer import Layer


class Linear(Layer):

    def __init__(self, input_num, output_num, bias=True):
        super().__init__()

        if bias:
            input_num += 1

        self.weights = np.zeros(output_num, input_num)

    def get_output(self, inp):
        self.input = inp
        self.output = T.dot(self.weights, self.input)
        return self.output

    def get_input_gradient(self, output_gradient):
        gradient = T.dot(self.weights.T, output_gradient)
        return gradient

    def get_parameter_gradient(self, output_gradient):
        pass

    def update_parameters(self, output_gradient, rate):
        self.weights = T.sub(self.weights, rate * self.get_input_gradient(output_gradient))

