import numpy as np
from network import NeuralNetwork
import pandas as pd
import os
import pickle
import warnings


# to supress numpy overflow warnings
warnings.filterwarnings('ignore')


LAYER_STRUCTURE = (784, 120, 10)


def formulate_data(data):
    # formulating the outputs and passing inputs and outputs to the main function
    outputs = np.zeros((data.shape[0], 10))
    for i in range(data[:, 0].shape[0]):
        outputs[i, data[i, 0]] = 1
    return data[:, 1:], outputs


def get_wb():
    w = None
    b = None
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "w_b.pickle"), "rb") as file:
        w, b = pickle.load(file)
    return w, b


def train(x, y):
    nn = NeuralNetwork(*LAYER_STRUCTURE)
    nn.grad_descent(x, y, output=True, epochs=1200)
    # comment the line below to stop updating the w_b.pickle everytime we train the model
    nn.save_data()


def test(x, y):
    print("Calculating accuracy...", end='')
    weights, biases = get_wb()
    accurate = 0
    nn = NeuralNetwork(*LAYER_STRUCTURE)
    for i in range(x.shape[0]):
        pred_y = nn.predict(x[i], w=weights, b=biases)
        if pred_y == np.argmax(y[i, :]):
            accurate = accurate + 1

    print(f'{round((accurate / x.shape[0]) * 100, 2)}%')


def predict(x):
    nn = NeuralNetwork(784, 60, 10)
    weights, biases = get_wb()
    prediction = nn.predict(x, w=weights, b=biases)
    print(f'Prediction is {prediction}')


if __name__ == "__main__":
    print("Preparing data...", end='')
    # getting the training data and converting to numpy array
    train_data = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "mnist_train.csv"),
                             header=None, skiprows=0).to_numpy()
    test_data = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "mnist_test.csv"),
                            header=None, skiprows=0).to_numpy()
    # shuffling the data
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    train_in, train_out = formulate_data(train_data)
    test_in, test_out = formulate_data(test_data)
    print("Done.")

    print("1. train the network")
    print("2. test the network")
    print("3. predict the digit")
    prog_choice = -1

    while True:
        try:
            prog_choice = int(input("Enter your choice: "))
            if prog_choice < 1 or prog_choice > 3:
                raise Exception("Please enter a number from 1 to 3. Try again.")
        except ValueError:
            print("Please enter a number from 1 to 3. Try again.")
        else:
            break

    if prog_choice == 1:
        train(train_in / 255, train_out)
    elif prog_choice == 2:
        test(test_in, test_out)
    else:
        print("Working on it...")
