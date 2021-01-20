# MNIST-Digit-Recognition
An artificial neural network to recognize handwritten digits, coded from scratch, in Python (MNIST Dataset).

This repository does not include the MNIST dataset because of its large size. But it is easily available [here](http://yann.lecun.com/exdb/mnist/). These files need to be extracted in the repository folder and then, "dataset_converter.py" will convert this dataset in a csv file, which will be placed in the "/data" directory. The code for this Python file is taken from [here](https://github.com/egcode/MNIST-to-CSV).

After the dataset files are extracted, it is safe to run "main.py" file. This file uses the "NeuralNetwork" class created inside "network.py" file.

For the reference, I have played with different values and created a "w_b.pickle" file, which updates the weights and biases every time we train the model. The current uploaded file gives 87.56% accuracy for the testing dataset. You can feel free to fiddle with the variables and get better accuracy from this model.
