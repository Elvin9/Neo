from lib.layers.layer import Layer
import gnumpy as gp


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):
        self.input = inp

        tmp_grt = gp.garray(inp) > gp.zeros(inp.shape)
        return gp.garray(inp) * tmp_grt

    def get_derivative(self):
        return gp.garray(self.input) > gp.zeros(self.input.shape)

    def get_input_gradient(self, output_gradient):
        return output_gradient * self.get_derivative()

