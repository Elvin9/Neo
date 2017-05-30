import gnumpy as gp


class SGD:

    def __init__(self, learning_rate):
        self.rate = learning_rate

    def parameters_delta(self, gradient):
        return -self.rate * gradient


class Momentum:

    def __init__(self, learning_rate, mu):
        self.v = None
        self.rate = learning_rate
        self.mu = mu

    def parameters_delta(self, gradient):
        if self.v is None:
            self.v = gp.zeros(gradient.shpe)

        self.v = (self.mu * self.v) - (self.rate * gradient)
        return self.v

