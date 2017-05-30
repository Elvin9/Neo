import gnumpy as gp

from lib.layers.layer import Layer


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):
        self.output = gp.logistic(inp)
        return self.output

    def get_derivative(self):
        sub_prod = 1 - self.output
        return self.output * sub_prod

    def get_input_gradient(self, output_gradient):
        return output_gradient * self.get_derivative()
