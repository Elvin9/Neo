from layer import Layer

from theano.tensor.nnet.nnet import sigmoid
import theano.tensor as T

import numpy as np

import math_utils

class Sigmoid(Layer):
	def __init__(self):
		super(self, Sigmoid)

	def get_output(self, input):
		self.output = sigmoid(input)
		return output

	def get_derivative(self):
		sub_prod = T.sub(np.ones([len(self.output), 1]), self.output)
		return math_utils.vec_elemwise(self.output, sub_prod)

	def get_input_gradient(self, output_gradient):
		return math_utils.vec_elemwise(output_gradient, get_derivative)