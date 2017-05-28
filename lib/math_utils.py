import theano.tensor as T
import numpy as np
from theano import function

# Element wise multiplication
x = T.dvector('x')
y = T.dvector('y')

z = x * y

vec_elemwise = function([x, y], z)

# ===========================
