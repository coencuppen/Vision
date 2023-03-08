import numpy as np
import matplotlib.pyplot as plt
import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.patches as patches
import matplotlib
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import cv2

import dataset

IMAGE_SHAPE = (224, 224)
bunniesArray = dataset.getPictures('bunnies', IMAGE_SHAPE)
randomPicturesArray = dataset.getPictures('randompictures', IMAGE_SHAPE)
print(bunniesArray.shape, randomPicturesArray.shape)
labels = []
for i in range(268):
    labels.append(1)
for i in range(262):
    labels.append(0)
labels = np.array(labels)

data = []
for i in bunniesArray:
    data.append(i)
for i in randomPicturesArray:
    data.append(i)

data = np.array(data)

print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=True)
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

IMAGE_SHAPE = IMAGE_SHAPE + (3,)
print(IMAGE_SHAPE)
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

model = tf.keras.Sequential([
    pretrained_model_without_top_layer,
    tf.keras.layers.Dense(2)
])

model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=5)


def evaluate(self):
    print(self.model.evaluate(self.testData, to_categorical(self.testLabels), verbose=2))
