import numpy as np
from lib.layers.linear import Linear
from lib.layers.sigmoid import Sigmoid
from lib.loss_functions.mse import MSELoss
from lib.models.sequential_model import SequentialModel
import matplotlib.pyplot as plt

rate = 1.5
data_size = 2000

model = SequentialModel(MSELoss(), rate)
model.add_layer(Linear(1, 4, bias=True))
model.add_layer(Sigmoid())
model.add_layer(Linear(4, 1, bias=True))

data_x = np.array([np.random.rand(data_size)*90])
data_x = np.deg2rad(data_x)
data_y = np.sin(data_x)

errors = model.train(data_x, data_y, batch_size=1, error=True)

print(errors)

test_x = np.arange(0, np.pi/2, 0.05)
test_y = np.sin(test_x)
pred = [model.forward(np.array([[x]])) for x in test_x]


plt.plot(test_x, test_y)
plt.scatter(test_x, pred)

plt.show()