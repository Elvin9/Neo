import gnumpy as gp
import numpy as np
from lib.layers.linear import Linear

__author__ = 'Alon'

layer = Linear(3, 2, bias=True)

inp = gp.garray([[1], [2], [3]])

# inp = gp.garray([1, 2, 3])

o = layer.get_output(inp)
d = layer.get_input_gradient(gp.garray([[1], [2],]))
print(o)

layer.update_parameters(gp.garray([[1], [2],]), 1)
# o = layer.get_output(inp)
# print(o)