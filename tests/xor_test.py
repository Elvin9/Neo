import numpy as np
from lib.layers.linear import Linear
from lib.layers.sigmoid import Sigmoid
from lib.loss_functions.mse import MSELoss
from lib.models.sequential_model import SequentialModel

rate = 100
data_size = 1000

loss = MSELoss()
model = SequentialModel(loss, rate)

model.add_layer(Linear(2, 2, bias=False))
model.add_layer(Sigmoid())
model.add_layer(Linear(2, 1, bias=False))
model.add_layer(Sigmoid())


x_data = np.array([np.random.binomial(1, 0.5, 2) for x in range(data_size)])
y_data = np.array([[x[0] ^ x[1] for x in x_data]])

x_data = x_data.T
y_data = y_data

errors = model.train(x_data, y_data, batch_size=1, error=True)
print(errors)
print('\n')

test_x1 = np.array([[0], [0]])
test_x2 = np.array([[1], [0]])
test_x3 = np.array([[0], [1]])
test_x4 = np.array([[1], [1]])


print(model.forward(test_x1))
print(model.forward(test_x2))
print(model.forward(test_x3))
print(model.forward(test_x4))

# model.add_layer(Linear(1, 1, bias=False))
#
# x_data = np.random.randint(10, size=(1, data_size))
# y_data = np.array(x_data)
#
# errors = model.train(x_data, y_data, batch_size=1, error=True)
# print(errors)
#
# test_x = np.array([[4]])
#
# print(model.forward(test_x))