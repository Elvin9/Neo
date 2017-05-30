import gnumpy as gp
from lib.layers.softmax import Softmax

softmax = Softmax()

a = gp.garray([[1, 3], [2, 2], [3, 1]])
o = softmax.get_output(a)
print(o)