from layer import Layer
from theano import function

from theano.tensor.nnet.nnet import sigmoid
import theano.tensor as T

import numpy as np

class Sigmoid(Layer):
	def __init__(self):
		self.output = 0

	def _get_output(input):
		self.output = sigmoid(input)
		return output

	def _get_derivative():
		sub_prod = T.sub(np.ones([len(self.output), 1]), self.output)

	def _get_input_gradient(output_gradient):
		return 