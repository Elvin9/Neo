import gnumpy as gp
import numpy as np
from lib.layers.linear import Linear

__author__ = 'Alon'

layer = Linear(3, 3, bias=False)

inp = gp.garray(np.array([[1,2,3]]).T)
print(inp)

o = layer.get_output(inp)
print(o)