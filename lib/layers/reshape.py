from lib.layers.layer import Layer


class Reshape(Layer):
    def __init__(self, out_shape):
        super().__init__()

        self.shape = out_shape
        self.input_shape = None

    def get_output(self, inp):
        self.input_shape = inp.shape
        return inp.reshape(self.shape)

    def get_input_gradient(self, output_gradient):
        return output_gradient.reshape(self.input_shape)

