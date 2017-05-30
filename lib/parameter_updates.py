import gnumpy as gp


class SGD:

    def __init__(self):
        pass

    def parameters_delta(self, gradient, rate):
        return -rate * gradient


class Momentum:

    def __init__(self, mu):
        self.v = None
        self.mu = mu

    def parameters_delta(self, gradient, rate):
        if self.v is None:
            self.v = gp.zeros(gradient.shpe)

        self.v = (self.mu * self.v) - (rate * gradient)
        return self.v

