from lib.layers.rectifiers import ReLU, LeakyReLU, ELU
import numpy as np

x = ReLU()

print(x.get_output(np.array([[1, -2, 3],
                            [-5, -3, 7]])))
print(x.get_derivative())

x = LeakyReLU()

print(x.get_output(np.array([[1, -2, 3],
                            [-5, -3, 7]])))
print(x.get_derivative())

x = ELU()

print(x.get_output(np.array([[1, -2, 3],
                            [-5, -3, 7]])))
print(x.get_derivative())
