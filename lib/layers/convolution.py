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

        self.kernel_weights = gp.randn(kernel_num, kernel_shape[0], kernel_shape[1], kernel_shape[2]) * \
                              np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2]))

        if self.bias:
            self.bias_weights = gp.zeros((kernel_num,))

        # self.kernel_weights = gp.garray([[[[1, 0, 0],
        #                                    [0, 1, 1],
        #                                    [1, 0, 0]],
        #
        #                                   [[1, 1, 1],
        #                                    [1, 1, 1],
        #                                    [1, 1, 1]],
        #              R.I.P
        #                                   [[1, 1, 1],
        #                                    [1, 1, 1],
        #                                    [1, 1, 1]]],
        #
        #                                  [[[1, 1, 1],
        #                                    [0, 1, 1],
        #                                    [1, 0, 0]],
        #           2017 - 2017
        #                                   [[1, 0, 1],
        #                                    [0, 1, 1],
        #                                    [1, 1, 1]],
        #
        #                                   [[1, 0, 1],
        #                                    [0, 0, 1],
        #                                    [1, 1, 1]]]])

    def get_output(self, inp):

        self.input = inp

        self.output = gp.zeros(self._get_output_shape(inp.shape))
        inp = self._zero_padding(inp, self.padding)

        # convert input to the same dimension as the kernels for the computation
        inp = inp.reshape((1, inp.shape[0], inp.shape[1], inp.shape[2]))

        for out_i, i in enumerate(range(0, inp.shape[2] - self.kernel_shape[1] + 1, self.stride)):
            for out_j, j in enumerate(range(0, inp.shape[3] - self.kernel_shape[2] + 1, self.stride)):
                field = inp[:, :, i:i + self.kernel_shape[1], j:j + self.kernel_shape[2]]
                # print(field)
                result = (self.kernel_weights.T * field.T).T
                sums = gp.sum(gp.sum(gp.sum(result, axis=1), axis=1), axis=1)

                if self.bias:
                    sums += self.bias_weights

                self.output[:, out_i, out_j] = sums

        return self.output

    def get_input_gradient(self, output_gradient):

        input_gradient = gp.zeros(self.input.shape)
        gradient_padding = (self.stride * (self.input.shape[1] - 1) - self.output.shape[1] + self.kernel_shape[1]) // 2
        output_gradient = self._zero_padding(output_gradient, gradient_padding)

        # flip the weights for the gradient computation
        weights = gp.garray(self.kernel_weights.as_numpy_array()[::, ::, ::-1, ::-1])

        for out_i, i in enumerate(range(0, output_gradient.shape[1] - self.kernel_shape[1] + 1, self.stride)):
            for out_j, j in enumerate(range(0, output_gradient.shape[2] - self.kernel_shape[2] + 1, self.stride)):
                field = output_gradient[:, i:i + self.kernel_shape[1], j:j + self.kernel_shape[2]]

                for weight_slice_k in range(self.kernel_shape[0]):
                    weight_slice = weights[:, weight_slice_k, :, :]
                    result = (weight_slice.T * field.T).T

                    sums = gp.sum(result)
                    input_gradient[weight_slice_k, out_i, out_j] = sums

        return input_gradient

    def get_parameter_gradient(self, output_gradient):

        gradient = gp.zeros(self.kernel_weights.shape)
        bias_gradient = None

        if self.bias:
            bias_gradient = gp.zeros(self.bias_weights.shape)

        inp = self._zero_padding(self.input, self.padding)

        # convert input to the same dimension as the kernels for the computation
        inp = inp.reshape((1, inp.shape[0], inp.shape[1], inp.shape[2]))

        for out_i, i in enumerate(range(0, inp.shape[2] - self.kernel_shape[1] + 1, self.stride)):
            for out_j, j in enumerate(range(0, inp.shape[3] - self.kernel_shape[2] + 1, self.stride)):
                field = inp[:, :, i:i + self.kernel_shape[1], j:j + self.kernel_shape[2]]

                for k in range(self.kernel_weights.shape[0]):
                    gradient[k] += field[0] * output_gradient[k, out_i, out_j]

                    if self.bias:
                        bias_gradient[k] += output_gradient[k, out_i, out_j]

        return gradient, bias_gradient

    def update_parameters(self, output_gradient, rate):
        gradient, bias_gradient = self.get_parameter_gradient(output_gradient)

        self.kernel_weights += self.parameter_update.parameters_delta(gradient, rate)

        if self.bias:
            self.bias_weights += self.parameter_update.parameters_delta(bias_gradient, rate)

    def _zero_padding(self, inp, padding):
        zeros = gp.zeros((inp.shape[0], padding, inp.shape[2]))
        inp = gp.concatenate((inp, zeros), axis=1)  # lower padding
        inp = gp.concatenate((zeros, inp), axis=1)  # upper padding

        zeros = gp.zeros((inp.shape[0], inp.shape[1], padding))
        inp = gp.concatenate((inp, zeros), axis=2)  # right padding
        inp = gp.concatenate((zeros, inp), axis=2)  # left padding

        return inp

    def _get_output_shape(self, inp_shape):

        assert inp_shape[0] == self.kernel_shape[0]

        shape = (self.kernel_num,
                 (((inp_shape[1] - self.kernel_shape[1] + self.padding * 2) // self.stride) + 1),
                 (((inp_shape[2] - self.kernel_shape[2] + self.padding * 2) // self.stride) + 1),
                 )

        return shape
