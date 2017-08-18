import gnumpy as gp
from lib.layers.layer import Layer


class MaxPooling(Layer):

    def __init__(self, kernel_num, pool_dim=(2, 2), stride=2):
        super().__init__()

        self.kernel_num = kernel_num
        self.pool_dim = pool_dim
        self.stride = stride
        self.input_gradient_template = None

    def get_output(self, inp):

        self.input_gradient_template = gp.zeros(inp.shape)  # create a ones and zeros tensor for later use in get_input_gradient
        self.output = gp.zeros(self._get_output_shape(inp.shape))

        # convert input to the same dimension as the kernels for the computation

        for out_i, i in enumerate(range(0, inp.shape[1] - self.pool_dim[0] + 1, self.stride)):
            for out_j, j in enumerate(range(0, inp.shape[2] - self.pool_dim[1] + 1, self.stride)):
                field = inp[:, i:i + self.pool_dim[0], j:j + self.pool_dim[1]]

                max_vals = field.max(axis=1).max(axis=1)

                self.output[:, out_i, out_j] = max_vals

                max_marks = (field.T < max_vals).T * -1 + 1  # mark with ones the maximums
                self.input_gradient_template[:, i:i + self.pool_dim[0], j:j + self.pool_dim[1]] = max_marks

        return self.output

    def get_input_gradient(self, output_gradient):

        template = self.input_gradient_template.copy()
        inp_grad = self.input_gradient_template

        for out_i, i in enumerate(range(0, inp_grad.shape[1] - self.pool_dim[0] + 1, self.stride)):
            for out_j, j in enumerate(range(0, inp_grad.shape[2] - self.pool_dim[1] + 1, self.stride)):
                field = template[:, i:i + self.pool_dim[0], j:j + self.pool_dim[1]]

                max_vals_gradients = output_gradient[:, out_i, out_j]

                # TODO fix this method, we don't like converting to numpy
                field_gradients = (field.as_numpy_array().T * max_vals_gradients.as_numpy_array()).T
                field_gradients = gp.garray(field_gradients)

                inp_grad[:, i:i + self.pool_dim[0], j:j + self.pool_dim[1]] = field_gradients

        return inp_grad

    def _get_output_shape(self, inp_shape):

        assert inp_shape[0] == self.kernel_num

        shape = (self.kernel_num,
                 (((inp_shape[1] - self.pool_dim[0]) // self.stride) + 1),
                 (((inp_shape[2] - self.pool_dim[1]) // self.stride) + 1),
                 )

        return shape
