import gnumpy as gp
from lib.layers.convolution import Convolution
from lib.parameter_updates import SGD

conv = Convolution((3, 3, 3), 2, SGD(), padding=1, bias=True)

a = gp.ones((3, 4, 4))
a = conv._zero_padding(a, 1)
# print(a)

inp = gp.garray([[[1, 0, 0],
                  [1, 0, 1],
                  [0, 1, 1]],

                 [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]],

                 [[0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0]]])

out = conv.get_output(inp)

# print(out)

output_grad = gp.garray([
                 [[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]],

                 [[0, 1, 0],
                  [1, 0, 0],
                  [0, 1, 0]],

                ])

grad = conv.get_input_gradient(output_grad)
print(grad)
conv.update_parameters(output_grad, rate=0.1)

# print(conv.bias_weights)
# print(conv.kernel_weights)
# print(grad)

