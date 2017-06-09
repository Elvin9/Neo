import gnumpy as gp
from lib.layers.convolution import Convolution

conv = Convolution((3, 3, 3), 1, None, padding=1)

a = gp.ones((3, 4, 4))
a = conv._zero_padding(a)
# print(a)

inp = gp.garray([[[1, 0, 0],
                  [1, 0, 1],
                  [0, 1, 1]],

                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]],

                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]])

out = conv.get_output(inp)

print(out)
