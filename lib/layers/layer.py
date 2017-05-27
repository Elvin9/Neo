import theano.tensor as T
from theano import function
import numpy as np


class Layer(object):
	def _get_output(input):
		raise NotImplemented

