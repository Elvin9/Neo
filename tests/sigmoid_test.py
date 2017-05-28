from lib.layers.sigmoid import Sigmoid
import numpy as np
import gnumpy as gp

x = Sigmoid()

print(x.get_output(gp.garray([[1, 2, 3],
                              [1, 2, 3]])))
print(x.get_derivative())
print(x.get_input_gradient(gp.garray([[2, 2, 3],
                                      [2, 2, 3]])))
