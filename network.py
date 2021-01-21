import numpy as np
import pickle
import os
from sys import stdout


class NeuralNetwork:
    def __init__(self, *size):
        self.n_layers = len(size)
        # initializing weights and biases
        self.w = [np.random.uniform(-0.12, 0.12, (m, n)) for m, n in zip(size[:-1], size[1:])]
        self.b = [np.random.uniform(-0.12, 0.12, (1, m)) for m in size[1:]]
        # inputs to each layers
        self.a = [[] for _ in range(self.n_layers)]
        # sigmoid outputs of each layers
        self.z = [[] for _ in range(self.n_layers - 1)]
        # gradients in weights and biases+
        self.grad_w = [[] for _ in range(self.n_layers - 1)]
        self.grad_b = [[] for _ in range(self.n_layers - 1)]
        # we will set this value to the number of inputs in a batch, later
        self.training_size = 0

    def forward(self, x, weights=None, biases=None):
        # weights and biases will be only passed if we are testing the model
        weights = self.w if weights is None else weights
        biases = self.b if biases is None else biases
        self.a[0] = x
        # setting the values for each inputs and outputs of layers
        for i in range(self.n_layers - 1):
            self.z[i] = self.a[i].dot(weights[i]) + biases[i]
            self.a[i+1] = self.sigmoid(self.z[i])
        return self.a[-1]

    def cost(self, y):
        j = -y * np.log(self.a[-1]) - (1 - y) * (1 - np.log(self.a[-1]))
        return np.sum(j) / self.training_size

    def back_prop(self, y):
        batch_size = float(len(y))
        delta = self.a[-1] - y
        # output layer weights and biases are modified using sigmoid of the outputs
        self.grad_w[-1] = (delta.T.dot(self.a[-2]) / batch_size).T
        self.grad_b[-1] = np.sum(delta, axis=0) / batch_size
        # hidden layer weights and biases are modified using derivative of the sigmoid
        if self.n_layers > 2:
            for i in range(2, self.n_layers):
                delta = delta.dot(self.w[-i+1].T) * self.sigmoid_deriv(self.z[-i])
                self.grad_w[-i] = (np.dot(delta.T, self.a[-i-1]) / batch_size).T
                self.grad_b[-i] = np.sum(delta, axis=0) / batch_size

    def grad_descent(self, x, y, epochs=500, learning_rate=0.08, output=False):
        self.training_size = 100
        count = 0
        for i in range(epochs):
            self.forward(x[count:count+self.training_size-1])
            self.back_prop(y[count:count+self.training_size-1])
            # update weights
            for l in range(self.n_layers - 1):
                self.w[l] = self.w[l] - learning_rate * self.grad_w[l]
                self.b[l] = self.b[l] - learning_rate * self.grad_b[l]
            # displaying output if required
            if output:
                stdout.write('\r%i/%i epochs completed.' % (i+1, epochs))
                stdout.flush()
            count = count + self.training_size
            if count + self.training_size - 1 > len(x):
                count = 0

    def predict(self, x, w=None, b=None):
        return np.argmax(self.forward(x, weights=w, biases=b), axis=1)[0]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))

    def save_data(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "w_b.pickle"), "wb") as file:
            pickle.dump((self.w, self.b), file)
