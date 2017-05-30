from lib.layers.layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def get_output(self, inp):
        self.input = inp

        if inp > 0:
            return inp
        return 0.01 * inp

    def get_derivative(self):
        if self.input > 0:
            return 1
        return 0.01

    def get_input_gradient(self, output_gradient):
        return output_gradient * self.get_derivative()

