from lib.loss_functions.mse import MSELoss
import gnumpy as gp

mse = MSELoss()

pred = gp.garray([[1, 1], [2, 2], [3, 3]])
output = gp.garray([[-1, -1], [1, 0], [2, 3]])

o = mse.get_output(output, pred)
d = mse.get_input_gradient(pred, output)

print(o)
print(d)

