import gnumpy as gp
import numpy as np
from lib.layers.layer import Layer


class Convolution(Layer):
    def __init__(self, kernel_shape, kernel_num, parameter_update, stride=1, padding=1, bias=True):
        super().__init__()

        self.bias = bias
        self.kernel_num = kernel_num
        self.parameter_update = parameter_update
        self.padding = padding
        self.stride = stride
        self.kernel_shape = kernel_shape

        # self.kernel_weights = gp.randn(kernel_num, kernel_shape[0], kernel_shape[1], kernel_shape[2]) * \
        #                       np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2]))

        self.kernel_weights = gp.garray([[[[1, 0, 0],
                                          [0, 1, 1],
                                          [1, 0, 0]],

                                         [[1, 1, 1],
                                          [1, 1, 1],
                                          [1, 1, 1]],

                                          [[1, 1, 1],
                                          [1, 1, 1],
                                          [1, 1, 1]]]])
                                         #
                                         #
                                         #
                                         # [[[1, 0, 0],
                                         #  [0, 1, 1],
                                         #  [1, 0, 0]],
                                         #
                                         # [[1, 1, 1],
                                         #  [1, 1, 1],
                                         #  [1, 1, 1]],
                                         #
                                         #  [[1, 1, 1],
                                         #  [1, 1, 1],
                                         #  [1, 1, 1]]]])

        # if self.bias:
        #     self.weights = gp.concatenate((self.weights, gp.zeros((output_num, 1))), axis=1)

    def get_output(self, inp):

        self.input = inp
        self.output = gp.zeros(self.get_output_shape(inp.shape))
        inp = self._zero_padding(inp)

        # convert input to the same dimension as the kernels for the computation
        inp = inp.reshape((1, inp.shape[0], inp.shape[1], inp.shape[2]))

        for out_i, i in enumerate(range(0, inp.shape[2] - self.kernel_shape[1] + 1, self.stride)):
            for out_j, j in enumerate(range(0, inp.shape[3] - self.kernel_shape[2] + 1, self.stride)):
                
                field = inp[:, :, i:i + self.kernel_shape[1], j:j + self.kernel_shape[2]]
                # print(field)
                result = (self.kernel_weights.T * field.T).T
                sums = gp.sum(gp.sum(gp.sum(result, axis=1), axis=1), axis=1)

                self.output[:, out_i, out_j] = sums

        return self.output

    def _zero_padding(self, inp):
        zeros = gp.zeros((inp.shape[0], self.padding, inp.shape[2]))
        inp = gp.concatenate((inp, zeros), axis=1)  # lower padding
        inp = gp.concatenate((zeros, inp), axis=1)  # upper padding

        zeros = gp.zeros((inp.shape[0], inp.shape[1], self.padding))
        inp = gp.concatenate((inp, zeros), axis=2)  # right padding
        inp = gp.concatenate((zeros, inp), axis=2)  # left padding

        return inp

    def get_output_shape(self, inp_shape):

        assert inp_shape[0] == self.kernel_shape[0]

        shape = (self.kernel_num,
                 (((inp_shape[1] - self.kernel_shape[1] + self.padding * 2) // self.stride) + 1),
                 (((inp_shape[2] - self.kernel_shape[2] + self.padding * 2) // self.stride) + 1),
                 )

        return shape
