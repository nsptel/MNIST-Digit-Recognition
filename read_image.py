import cv2 as cv
import numpy as np
import os


def read_file(path):
    img = cv.resize(cv.imread(path, cv.IMREAD_GRAYSCALE), dsize=(28, 28))
    _, bw = cv.threshold(img, 120, 255, cv.THRESH_BINARY_INV)
    bw = np.reshape(bw, (784, ))
    return bw


if __name__ == "__main__":
    print(read_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "test_images", "digit (9).jpg")))
