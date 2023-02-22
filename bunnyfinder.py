import numpy as np
import matplotlib.pyplot as plt
import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar10


class bunnyfinder:
    def __init__(self):
        self.model = Sequential([])

        self.model.compile()

        self.model.fit()

    def evaluate(self):
        #print(self.model.evaluate(test_images,  to_categorical(test_labels), verbose=2))
        print("evaluate")
