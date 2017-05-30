import gnumpy as gp
from lib.loss_functions.cross_entropy import CrossEntropy

loss = CrossEntropy()

target = gp.garray([[0, 1], [0, 0 ], [1, 1]])
output = gp.garray([[0.3, 0.4], [0.3, 0.3], [0.4, 0.4]])

o = loss.get_output(output, target)
d = loss.get_input_gradient(target)

print(o)
print(d)