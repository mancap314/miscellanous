import numpy as np

def nonlin(x, deriv=False):
    '''sigmoid function'''
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

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

for _ in range(10000):
     l1 = nonlin(np.dot(X, syn0)) #forward propagation
     l1_error = y - l1 #value error
     l1_delta = np.multiply(l1_error, nonlin(l1, True)) #value error * slope of the error function at the value
     syn0 += np.dot(X.T, l1_delta) #updating of the weights

print('Output after training:\n{}'.format(l1))
