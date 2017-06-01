import random
import numpy as np
import matplotlib.pyplot as plt
from lib.layers.linear import Linear
from lib.layers.softmax import SoftmaxCrossEntropyLayer
from lib.layers.tanh import Tanh
from lib.loss_functions.cross_entropy import SoftmaxCrossEntropyLoss
from lib.loss_functions.mse import MSELoss
from lib.models.sequential_model import SequentialModel
from lib.parameter_updates import SGD, Momentum
from lib.preprocessing import mean_subtraction, normalization

test_train_ratio = 1.0 / 5


def parse_line(line):
    x = []
    chars = line.strip('\n').split(",")
    for c in chars[1:]:
        x += [ord(c)]

    y = ord(chars[0]) == ord('p')
    return np.array([x]).T, np.array([[float(y)], [float(not y)]])


data_file = open("data.txt", mode='r').readlines()
random.shuffle(data_file)

x_data, y_data = parse_line(data_file[0])

# parse the data file
for line in data_file[1:]:
    x, y = parse_line(line)
    y_data = np.concatenate((y_data, y), axis=1)
    x_data = np.concatenate((x_data, x), axis=1)


data_size = len(x_data[0])

# normalize the data
x_data = mean_subtraction(x_data)
x_data = normalization(x_data)

# split the data
test_part = int(test_train_ratio * data_size)
# test_part = data_size - 4500

x_train = np.array(x_data[:, test_part:])
y_train = np.array(y_data[:, test_part:])

x_test = np.array(x_data[:, :test_part])
y_test = np.array(y_data[:, :test_part])


# ============ Training done here! =============

# -------------- 96 % model ------------
# rate = 0.007
# model = SequentialModel(rate, MSELoss())
# model.add_layer(Linear(22, 30, parameter_update=SGD()))
# model.add_layer(Tanh())
# model.add_layer(Linear(30, 2, parameter_update=SGD()))
# model.add_layer(Tanh())

# ------------ 97 % model ---------------
# rate = 0.009
# model = SequentialModel(rate, MSELoss())
# model.add_layer(Linear(22, 30, parameter_update=Momentum()))
# model.add_layer(Tanh())
# model.add_layer(Linear(30, 30, parameter_update=Momentum()))
# model.add_layer(Tanh())
# model.add_layer(Linear(30, 2, parameter_update=Momentum()))
# model.add_layer(Tanh())

# ----------- 99 % model !!! the BEST --------
rate = 0.001
model = SequentialModel(rate, SoftmaxCrossEntropyLoss())
model.add_layer(Linear(22, 30, parameter_update=Momentum()))
model.add_layer(Tanh())
model.add_layer(Linear(30, 30, parameter_update=Momentum()))
model.add_layer(Tanh())
model.add_layer(Linear(30, 2, parameter_update=Momentum()))
model.add_layer(SoftmaxCrossEntropyLayer())

errors = model.train(x_train, y_train, batch_size=20, error=True)

correct = 0.0


for i in range(x_test.shape[1]):
    case = np.array([x_test[:, i]]).T
    pred = model.forward(case)
    target = y_test[:, i]
    # print(pred)

    if pred[0] > 0.5 and target[0] == 1:
        correct += 1

    elif pred[0] < 0.5 and target[0] == 0:
        correct += 1

print("[*] Test data size: {}".format(x_test.shape[1]))
print("[*] Test Result: {}".format(correct / x_test.shape[1]))

error_x = np.arange(0, len(errors), 1)
plt.plot(error_x, errors)

plt.show()

