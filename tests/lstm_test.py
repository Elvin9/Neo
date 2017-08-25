import gnumpy as gp
from lib.layers.lstm import LSTM
from lib.parameter_updates import SGD

lstm = LSTM(2, 1, parameter_update=SGD(), return_sequence=True)

a = gp.garray([[1],
               [2]])

b = gp.garray([[0.5],
               [3]])

c = gp.garray([[0.2],
               [4]])

# out_grad = gp.garray([[0.2]])
out_grad = [gp.garray([[0.2]]), gp.garray([[0.34]]), gp.garray([[0.563]])]

seq = [a, b, c]

out = lstm.get_output(seq)
print(out)

inp_grad = lstm.get_input_gradient(out_grad)
print(inp_grad)

lstm.update_parameters(None, 0.1)

out = lstm.get_output(seq)
print(out)