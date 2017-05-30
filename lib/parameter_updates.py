import gnumpy as gp

class ParameterUpdate:

    def parameters_delta(self, gradient, rate):
        pass


class SGD(ParameterUpdate):
    def __init__(self):
        pass

    def parameters_delta(self, gradient, rate):
        return -rate * gradient


class Momentum(ParameterUpdate):
    """
    mu is usually something like: [0.5, 0.9, 0.95, 0.99]
    """

    def __init__(self, mu=0.9):
        self.v = None
        self.mu = mu

    def parameters_delta(self, gradient, rate):
        if self.v is None:
            self.v = gp.zeros(gradient.shape)

        self.v = (self.mu * self.v) - (rate * gradient)
        return self.v


class Adagrad(ParameterUpdate):
    """
    epsilon is usually somewhere between: [1e-4, 1e-8]
    """

    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon
        self.cache = None

    def parameters_delta(self, gradient, rate):
        if self.cache is None:
            self.cache = gp.zeros(gradient.shape)

        self.cache += gradient * gradient
        denumer = gp.sqrt(self.cache) + self.epsilon
        inv_denumer = gp.garray(1.0 / denumer.as_numpy_array())
        return (-rate * gradient) * inv_denumer
