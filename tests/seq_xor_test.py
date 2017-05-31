import numpy as np

from lib.layers.rectifiers import ReLU
from lib.layers.linear import Linear
from lib.layers.sigmoid import Sigmoid
from lib.layers.softmax import Softmax
from lib.layers.tanh import Tanh
from lib.loss_functions.mse import MSELoss
from lib.loss_functions.cross_entropy import CrossEntropy
from lib.models.sequential_model import SequentialModel
from lib.parameter_updates import SGD, Adagrad, Momentum
import matplotlib.pyplot as plt

rate = 0.001
data_size = 10000

model = SequentialModel(rate, MSELoss())

model.add_layer(Linear(2, 5, bias=False, parameter_update=Momentum()))
model.add_layer(Tanh())
model.add_layer(Linear(5, 5, bias=False, parameter_update=Momentum()))
model.add_layer(Tanh())
model.add_layer(Linear(5, 2, bias=False, parameter_update=Momentum()))


x_data = np.array([np.random.binomial(1, 0.5, 2) for x in range(data_size)])
y_data = np.array([[x[0] ^ x[1] for x in x_data], [not(x[0] ^ x[1]) for x in x_data]])

x_data = x_data.T
y_data = y_data


errors = model.train(x_data, y_data, batch_size=10, error=True)
print('\n')

test_x = np.array([[1], [0]])
print("1  0 : " + str(model.forward(test_x)))

test_x = np.array([[0], [1]])
print("0  1 : " + str(model.forward(test_x)))

test_x = np.array([[1], [1]])
print("1  1 : " + str(model.forward(test_x)))

test_x = np.array([[0], [0]])
print("0  0 : " + str(model.forward(test_x)))

error_x = np.arange(0, len(errors), 1)
plt.plot(error_x, errors)

plt.show()



