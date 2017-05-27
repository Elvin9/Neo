import theano.tensor as T
import numpy as np
from theano import function

x = T.dvector('x')
y = T.dvector('y')
z = x * y
vec_elemwise = function([x, y], z)

a = np.array([1,2,3])
b = np.array([1,2,3])
print(vec_elemwise(a, b))