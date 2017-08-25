import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from lib.layers.convolution import Convolution
from lib.layers.linear import Linear
from lib.layers.pooling import MaxPooling
from lib.layers.rectifiers import ReLU
from lib.layers.reshape import Reshape
from lib.layers.softmax import SoftmaxCrossEntropyLayer
from lib.loss_functions.cross_entropy import SoftmaxCrossEntropyLoss
from lib.models.model_loading import *
from lib.models.sequential_model import SequentialConvolutionalModel
from lib.parameter_updates import SGD, Momentum
from lib.preprocessing import *

image_shape = (1, 28, 28)
data = MNIST('./data')
images, labels = data.load_training()

x_train = [np.array(m).reshape(image_shape) for m in images]
y_train = []

for l in labels:
    z = np.zeros(10)
    z[l] = 1
    y_train.append(z.reshape(10, 1))

y_train = y_train
x_train = np.array(list(map(mean_subtraction, x_train)))
x_train = normalization(x_train)

rate = 0.001

# x_train = x_train[1200:1600]
# y_train = y_train[1200:1600]

model = load_model('mnist-conv.neom')

if model is None:
    print("Creating the model...")
    model = SequentialConvolutionalModel(rate, SoftmaxCrossEntropyLoss())
    model.add_layer(Convolution((1, 5, 5), 8, Momentum(), padding=2))
    model.add_layer(ReLU())
    model.add_layer(MaxPooling(8))
    model.add_layer(Convolution((8, 5, 5), 16, Momentum(), padding=2))
    model.add_layer(ReLU())
    model.add_layer(MaxPooling(16))
    model.add_layer(Reshape((784, 1)))
    model.add_layer(Linear(784, 10, Momentum()))
    model.add_layer(SoftmaxCrossEntropyLayer())

print(model.forward(x_train[1807]))
print(y_train[1807])
# errors = model.train(x_train, y_train, batch_size=1, error=True)
#
# save_model(model, 'mnist-conv.neom')
#
# error_x = np.arange(0, len(errors), 1)
# plt.plot(error_x, errors)
#
# # plt.matshow(x_train[1].reshape(28,28))
# # plt.text(2, 2, str(labels[1]), fontsize=12)
# plt.show()
