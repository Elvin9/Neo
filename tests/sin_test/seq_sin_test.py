import numpy as np
from lib.layers.linear import Linear
from lib.layers.sigmoid import Sigmoid
from lib.loss_functions.mse import MSELoss
from lib.models.sequential_model import SequentialModel
import matplotlib.pyplot as plt
from lib.parameter_updates import SGD, Momentum

rate = 0.09
mu = 0.8
data_size = 50000

model = SequentialModel(rate, MSELoss())

model.add_layer(Linear(1, 10, parameter_update=Momentum(mu), bias=True))
model.add_layer(Sigmoid())
model.add_layer(Linear(10, 1, parameter_update=Momentum(mu), bias=True))

data_x = np.array([np.random.rand(data_size)*180])
data_x = np.deg2rad(data_x)
data_y = np.sin(data_x)

errors = model.train(data_x, data_y, batch_size=5, error=False)

# print(errors)

test_x = np.arange(0, np.pi, 0.05)
test_y = np.sin(test_x)
pred = [model.forward(np.array([[x]])) for x in test_x]


plt.plot(test_x, test_y)
plt.scatter(test_x, pred)

plt.show()
