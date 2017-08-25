import gnumpy as gp
import numpy as np
from .layer import Layer


class Linear(Layer):

    def __init__(self, input_num, output_num, parameter_update, bias=True):
        super().__init__()

        self.bias = bias
        self.parameter_update = parameter_update

        self.weights = (gp.randn(output_num, input_num) * (np.sqrt(2.0 / input_num)))

        if self.bias:
            self.weights = gp.concatenate((self.weights, gp.zeros((output_num, 1))), axis=1)

    def get_output(self, inp):

        if self.bias:
            self.input = gp.concatenate((inp, gp.ones((1, inp.shape[1]))))
        else:
            self.input = inp

        self.output = gp.dot(self.weights, self.input)
        return self.output

    def get_input_gradient(self, output_gradient):

        w_t = self.weights.T

        if self.bias:
            if gp._useGpu == 'no':
                w_t = w_t.as_numpy_array()
                w_t = gp.garray(np.reshape(w_t[:len(w_t) - 1], [w_t.shape[0] - 1, w_t.shape[1]]))
            if gp._useGpu == 'yes':
                w_t = w_t[:len(w_t) - 1]

            gradient = gp.dot(w_t, output_gradient)
        else:
            gradient = gp.dot(w_t, output_gradient)
        return gradient

    def get_parameter_gradient(self, output_gradient):
        gradient = gp.dot(output_gradient, self.input.T)
        return gradient

    def update_parameters(self, output_gradient, rate, parameter_gradient=None):

        if parameter_gradient is None:
            gradient = self.get_parameter_gradient(output_gradient)
        else:
            gradient = parameter_gradient

        self.weights += self.parameter_update.parameters_delta(gradient, rate)


