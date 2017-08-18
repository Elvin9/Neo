import gnumpy as gp
from lib.layers.pooling import MaxPooling

pooling = MaxPooling(3)

inp = gp.garray([[[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9,10,11,12],
                  [13,14,15,16]],

                 [[22, 2, 22, 4],
                  [5, 6, 7, 8],
                  [9,10,11,12],
                  [22,14,22,16]],

                 [[22, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9,10,11,12],
                  [13,14,15,16]]])

out_grad = gp.garray([[[1, 2],
                       [3, 4]],
                      [[5, 6],
                       [7, 8]],
                      [[9, 10],
                       [11, 12]]])

print(pooling.get_output(inp))
print(pooling.input_gradient_template)
print(pooling.get_input_gradient(out_grad))
