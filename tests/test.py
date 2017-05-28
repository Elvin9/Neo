import numpy as np
import gnumpy as gp
import time

size = 1000

na = np.random.randn(size, size)
nb = np.random.randn(size, size)

start_t = time.time()
a = gp.garray(na)
b = gp.garray(nb)
end_t = time.time()

print('Initialization time: %f' % (end_t - start_t))

for i in range(4):

    start_t = time.time()
    res = na * nb
    end_t = time.time()

    print('Numpy time: %f' % (end_t - start_t))

    start_t = time.time()
    res = a * b
    end_t = time.time()

    print('GPU time: %f' % (end_t - start_t))

    print("\n------------------------\n")

