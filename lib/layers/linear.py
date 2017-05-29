import gnumpy as gp
from .layer import Layer


class Linear(Layer):

    def __init__(self, input_num, output_num, bias=True):
        super().__init__()

        self.bias = bias

        if self.bias:
            input_num += 1

        self.weights = (gp.rand(output_num, input_num) * (1.0 / gp.sqrt(input_num)))

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
            gradient = gp.dot(w_t[:len(w_t) - 1], output_gradient)
        else:
            gradient = gp.dot(w_t, output_gradient)
        return gradient

    def get_parameter_gradient(self, output_gradient):
        gradient = gp.dot(output_gradient, self.input.T)
        return gradient

    def update_parameters(self, output_gradient, rate):
        self.weights = self.weights - (rate * self.get_parameter_gradient(output_gradient))

