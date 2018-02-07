'''toy full batch neural network, largely inspired by http://iamtrask.github.io/2015/07/12/basic-python-network/'''

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1) #set seed (for reproducibility)


def nonlin(x, deriv=False):
    '''sigmoid function'''
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def lin(x, deriv=False):
    '''linear function'''
    is_in = (np.abs(x) <= 1).astype(float)
    if deriv:
        return is_in
    return 0.5 * (is_in * x  + (1 - is_in) * np.sign(x) + 1)


def plot_error(sq_errs):
    lgds = []
    for name, sq_err in sq_errs.items():
        plt.plot(sq_err, label=name)
        lgds.append(name)
    plt.legend(lgds)
    plt.ylabel('Squared Error')
    plt.ylim((0, max([max(sq_err) for sq_err in sq_errs.values()]) + 0.05))
    plt.xlabel('epoch')
    plt.title('Evolution of Squared Error by Epoch')
    plt.show()


def train_1(X, y, activation_function, n_epoch=1000, initial_weights_scaling=0.1):
    '''train a 1-layer nn'''
    # initialize weights
    syn0 = initial_weights_scaling * np.random.random((3, 1)) - initial_weights_scaling / 2 #1st term in (0, 2), -1: in (-1, 1), mean: 0
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


def train_2(X, y, activation_function, n_epoch=60000, initial_weights_scaling=0.1):
    '''train a 2-layer nn'''
    syn0 = initial_weights_scaling * np.random.random((3, 4)) - initial_weights_scaling / 2
    syn1 = 2 * np.random.random((4, 1)) - 1
    sq_err = []
    for i in range(n_epoch):  # push forward
        l1 = activation_function(X.dot(syn0))
        l2 = activation_function(l1.dot(syn1))

        # l2
        l2_error = y - l2
        l2_delta = l2_error * activation_function(l2, True)

        sq_err.append(np.square(l2_error).mean())

        # l1, back-propagation
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * activation_function(l1, True)

        # weight update
        syn1 += l1.T.dot(l2_delta)  # correction = input * delta
        syn0 += X.T.dot(l1_delta)

    return l2, sq_err


def run_training(X, y, train, n_epoch=1000, initial_weights_scaling=0.5):
    activation_functions = {'non-linear': nonlin, 'linear':  lin}
    sq_errs = {}
    for name, activation_function in activation_functions.items():
        l1, sq_err = train(X, y, activation_function, n_epoch)
        print('{}: output after training:\n{}'.format(name, l1))
        sq_errs[name] = sq_err

    plot_error(sq_errs)


def run_example_1(n_epoch=1000, initial_weights_scaling=0.5):
    X = np.array([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 0, 1]])
    y = np.array([[0, 0, 1, 1]]).T
    run_training(X, y, train_1, n_epoch, initial_weights_scaling)


def run_example_2(n_epoch=1000, initial_weights_scaling=0.5):
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0, 1, 1, 0]]).T
    run_training(X, y, train_2, n_epoch, initial_weights_scaling)

run_example_1()
run_example_2(n_epoch=60000, initial_weights_scaling=0.01)
