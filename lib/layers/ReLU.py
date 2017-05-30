from lib.layers.layer import Layer
import gnumpy as gp


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):
        self.input = inp

        tmp_grt = gp.garray(inp) > 0
        return gp.garray(inp) * tmp_grt

    def get_derivative(self):
        return gp.garray(self.input) > gp.zeros(self.input.shape)

    def get_input_gradient(self, output_gradient):
        return output_gradient * self.get_derivative()


class LeakyReLU(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):
        self.input = inp

        inp = gp.garray(inp)

        tmp_grt = inp < 0
        tmp_grt2 = inp > 0

        return tmp_grt * 0.01 * inp + tmp_grt2 * inp

    def get_derivative(self):
        return gp.garray(self.input) > gp.zeros(self.input.shape)

    def get_input_gradient(self, output_gradient):
        return output_gradient * self.get_derivative()


class ELU(Layer):
    def __init__(self, coefficient=0.01):
        super().__init__()
        self.coefficient = coefficient

    def get_output(self, inp):
        self.input = inp

        inp = gp.garray(inp)

        tmp_grt = inp < 0
        tmp_grt2 = inp > 0

        coefficient = self.coefficient * (gp.exp(inp) - gp.ones(inp.shape))

        return tmp_grt * coefficient * inp + tmp_grt2 * inp

    def get_derivative(self):
        return gp.garray(self.input) > gp.zeros(self.input.shape)

    def get_input_gradient(self, output_gradient):
        return output_gradient * self.get_derivative()

