import gnumpy as gp
import copy

from lib.layers.tanh import Tanh
from lib.layers.layer import Layer
from lib.layers.linear import Linear
from lib.layers.sigmoid import Sigmoid


class LSTM(Layer):
    def __init__(self, input_num, output_num, parameter_update=None, bias=True, return_sequence=False):
        super().__init__()

        self.input_num = input_num
        self.output_num = output_num
        self.bias = bias
        self.return_sequence = return_sequence

        self.i_linear = Linear(input_num + output_num, output_num, copy.deepcopy(parameter_update))
        self.i_sigm = Sigmoid()

        self.f_linear = Linear(input_num + output_num, output_num, copy.deepcopy(parameter_update))
        self.f_sigm = Sigmoid()

        self.o_linear = Linear(input_num + output_num, output_num, copy.deepcopy(parameter_update))
        self.o_sigm = Sigmoid()

        self.g_linear = Linear(input_num + output_num, output_num, copy.deepcopy(parameter_update))
        self.g_tanh = Tanh()

        self.c_tanh = Tanh()

        # gate_cache is a list of tuples, that looks like [(i, f, o, g, prev_ct, ct, x), ...]
        self.step_cache = []

        self.i_param_grad = None
        self.f_param_grad = None
        self.o_param_grad = None
        self.g_param_grad = None

    def get_output(self, inp):

        self.input = inp
        batch_size = self.input[0].shape[1]

        if self.return_sequence:
            self.output = []

        prev_ht = gp.zeros((self.output_num, batch_size))
        prev_ct = gp.zeros((self.output_num, batch_size))

        # clear the gate_cache before the forward unrolling
        self.step_cache = []

        # unrolling through time
        for index, inp_t in enumerate(self.input):
            ht, ct = self._lstm_forward_step(prev_ht, prev_ct, inp_t)

            if self.return_sequence:
                self.output.append(ht)
            else:
                self.output = ht

            prev_ht, prev_ct = ht, ct

        return self.output

    def _lstm_forward_step(self, prev_ht, prev_ct, inp_t):

        x = gp.concatenate((inp_t, prev_ht))

        i = self.i_sigm.get_output(self.i_linear.get_output(x))
        f = self.f_sigm.get_output(self.f_linear.get_output(x))
        o = self.o_sigm.get_output(self.o_linear.get_output(x))
        g = self.g_tanh.get_output(self.g_linear.get_output(x))

        ct = (f * prev_ct) + (i * g)
        ht = o * self.c_tanh.get_output(ct)

        # print(i, f, o, g, ct, ht)
        # print("----------------")

        # save the step state to the cache (save the input with the bias)
        self.step_cache.append((i, f, o, g, prev_ct, ct, self.i_linear.input))
        return ht, ct

    def get_input_gradient(self, output_gradient):

        # clear the previous parameter gradient
        self.i_param_grad = None
        self.f_param_grad = None
        self.o_param_grad = None
        self.g_param_grad = None

        input_gradient = None

        if self.return_sequence:

            for i in reversed(range(len(self.input))):
                current_out_grad = output_gradient[i]
                current_inp_grad = self._lstm_bptt(current_out_grad, i + 1)

                if input_gradient is None:
                    input_gradient = current_inp_grad
                else:
                    for j in range(len(current_inp_grad)):
                        input_gradient[i] += current_inp_grad[i]

        else:
            input_gradient = self._lstm_bptt(output_gradient, len(self.input))

        return input_gradient

    def _lstm_bptt(self, output_gradient, current_length):
        input_gradient = []

        ht_grad = output_gradient
        ct_grad = gp.zeros(ht_grad.shape)
        inp_grad = None

        for i in reversed(range(current_length)):
            prev_ht_grad, prev_ct_grad, inp_grad = self._lstm_backward_step(ht_grad, ct_grad, self.step_cache[i])

            input_gradient.append(inp_grad)

            ht_grad = prev_ht_grad
            ct_grad = prev_ct_grad

        # reverse the order to match the input's order
        input_gradient.reverse()

        return input_gradient

    def _lstm_backward_step(self, ht_grad, ct_grad, gate_cache):

        # unpack the state at time t from the cache
        i, f, o, g, prev_ct, ct, x = gate_cache

        o_grad = ht_grad * self.c_tanh.get_output(ct)
        ct_grad += ht_grad * o * self.c_tanh.get_input_gradient(1)

        # calculate the gradient of each gate
        prev_ct_grad = f * ct_grad
        i_grad = ct_grad * g
        f_grad = ct_grad * prev_ct
        g_grad = ct_grad * i

        # set the layers to their state on time t
        self.i_sigm.output = i
        self.f_sigm.output = f
        self.o_sigm.output = o
        self.g_tanh.output = g

        self.i_linear.input = x
        self.f_linear.input = x
        self.o_linear.input = x
        self.g_linear.input = x

        # calculate input gradients
        i_inp_grad = self.i_linear.get_input_gradient(self.i_sigm.get_input_gradient(i_grad))
        f_inp_grad = self.f_linear.get_input_gradient(self.f_sigm.get_input_gradient(f_grad))
        o_inp_grad = self.o_linear.get_input_gradient(self.o_sigm.get_input_gradient(o_grad))
        g_inp_grad = self.g_linear.get_input_gradient(self.g_tanh.get_input_gradient(g_grad))

        # split the input gradient to the real input and prev_h
        inp_grad = i_inp_grad[:self.input_num] + f_inp_grad[:self.input_num] + \
                   o_inp_grad[:self.input_num] + g_inp_grad[:self.input_num]

        prev_ht_grad = i_inp_grad[self.input_num:] + f_inp_grad[self.input_num:] + \
                       o_inp_grad[self.input_num:] + g_inp_grad[self.input_num:]

        # update the parameter gradient for each gate
        if self.i_param_grad is None:
            self.i_param_grad = self.i_linear.get_parameter_gradient(i_grad)
            self.f_param_grad = self.f_linear.get_parameter_gradient(f_grad)
            self.o_param_grad = self.o_linear.get_parameter_gradient(o_grad)
            self.g_param_grad = self.g_linear.get_parameter_gradient(g_grad)
        else:
            self.i_param_grad += self.i_linear.get_parameter_gradient(i_grad)
            self.f_param_grad += self.f_linear.get_parameter_gradient(f_grad)
            self.o_param_grad += self.o_linear.get_parameter_gradient(o_grad)
            self.g_param_grad += self.g_linear.get_parameter_gradient(g_grad)

        return prev_ht_grad, prev_ct_grad, inp_grad

    def update_parameters(self, output_gradient, rate):

        # the argument output_gradient can be None because we specify the parameter gradient
        self.i_linear.update_parameters(None, rate, parameter_gradient=self.i_param_grad)
        self.f_linear.update_parameters(None, rate, parameter_gradient=self.f_param_grad)
        self.o_linear.update_parameters(None, rate, parameter_gradient=self.o_param_grad)
        self.g_linear.update_parameters(None, rate, parameter_gradient=self.g_param_grad)
