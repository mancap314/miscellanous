'''toy full batch neural network, largely inspired by http://iamtrask.github.io/2015/07/12/basic-python-network/'''

import numpy as np
import matplotlib.pyplot as plt


def nonlin(x, deriv=False):
    '''sigmoid function'''
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def lin(x, deriv=False):
    ''' linear function'''
    is_in = (np.abs(x) <= 1).astype(float)
    if deriv:
        return is_in
    return 0.5 * (is_in * x  + (1 - is_in) * np.sign(x) + 1)


def plot_error(sq_errs):
    lgds = []
    for name, sq_err in sq_errs.items():
        line = plt.plot(sq_err, label=name)
        lgds.append(name)
    plt.legend(lgds)
    plt.ylabel('Squared Error')
    plt.ylim((0, max([max(sq_err) for sq_err in sq_errs.values()]) + 0.05))
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

def train(activation_function, n_epoch=1000):
    # initialize weights
    syn0 = 2 * np.random.random((3, 1)) - 1 #1st term in (0, 2), -1: in (-1, 1), mean: 0
    sq_err = []
    for i in range(n_epoch):
         l1 = activation_function(np.dot(X, syn0)) #forward propagation
         if (i + 1) % 100 == 0:
             print('Epoch {}, l1=\n{}'.format(i+1, l1))
         l1_error = y - l1 #value error
         sq_err.append(np.square(l1_error).mean())
         l1_delta = np.multiply(l1_error, nonlin(l1, True)) #value error * slope of the error function at the value
         syn0 += np.dot(X.T, l1_delta) #updating of the weights
    return l1, sq_err

activation_functions = {'non-linear': nonlin, 'linear':  lin}
sq_errs = {}
for name, activation_function in activation_functions.items():
    l1, sq_err = train(activation_function, 1000)
    print('{}: output after training:\n{}'.format(name, l1))
    sq_errs[name] = sq_err

plot_error(sq_errs)

# linear learns faster when it learns...
