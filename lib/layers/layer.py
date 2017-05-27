class Layer(object):
    def __init__(self):
        self.output = 0
        self.input = 0

    def get_output(self, inp):
        raise NotImplemented

    def update_parameters(self, delta):
        raise NotImplemented

    def get_input_gradient(self, output_gradient):
        raise NotImplemented
