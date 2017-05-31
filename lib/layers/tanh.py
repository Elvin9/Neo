import gnumpy as gp

from lib.layers.layer import Layer


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):
        self.output = gp.tanh(inp)
        return self.output

    def get_derivative(self):
        return 1 - (self.output * self.output)

    def get_input_gradient(self, output_gradient):
        return output_gradient * self.get_derivative()
