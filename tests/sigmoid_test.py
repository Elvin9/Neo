from lib.layers.sigmoid import Sigmoid
import numpy as np

x = Sigmoid()

print(x.get_output(np.array([1, 2, 3])).eval())
print(x.get_derivative().eval())
