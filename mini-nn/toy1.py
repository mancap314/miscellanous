'''toy full batch neural network, largely inspired by http://iamtrask.github.io/2015/07/12/basic-python-network/'''

import numpy as np
import matplotlib.pyplot as plt

def nonlin(x, deriv=False):
    '''sigmoid function'''
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def plot_error(sq_err):
   plt.plot(sq_err)
   plt.ylabel('Squared Error')
   plt.ylim((0, max()))
   plt.xlabel('epoch')
   plt.title('Evolution of Squared Error by Epoch')
   plt.show()

#Input
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 0, 1]])

# value to find aka ground truth aka...
y = np.array([[0, 0, 1, 1]]).T

# set seed (for reproducibility)
np.random.seed(1)

# initialize weights
syn0 = 2 * np.random.random((3, 1)) - 1 #1st term in (0, 2), -1: in (-1, 1), mean: 0
sq_err = []
for _ in range(100):
     l1 = nonlin(np.dot(X, syn0)) #forward propagation
     l1_error = y - l1 #value error
     sq_err.append(np.square(l1_error).mean())
     l1_delta = np.multiply(l1_error, nonlin(l1, True)) #value error * slope of the error function at the value
     syn0 += np.dot(X.T, l1_delta) #updating of the weights

print('Output after training:\n{}'.format(l1))
plot_error(sq_err)
