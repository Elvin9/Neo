from lib.layers.ReLU import ReLU
import numpy as np

x = ReLU()

print(x.get_output(np.array([[1, -2, 3],
                            [-5, -3, 7]])))
print(x.get_derivative())
