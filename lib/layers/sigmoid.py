from theano.tensor.nnet.nnet import sigmoid
import theano.tensor as T

import numpy as np

from lib import math_utils
from lib.layers.layer import Layer


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):
        self.output = sigmoid(inp)
        return self.output

    def get_derivative(self):
        sub_prod = T.sub(np.ones(self.output.shape.eval()[0]), self.output)
        return self.output * sub_prod

    def get_input_gradient(self, output_gradient):
        return math_utils.vec_elemwise(output_gradient, self.get_derivative())
