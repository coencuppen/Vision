import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image
from scipy import ndimage
import scipy
import os, sys
import json


jsonFile = open('coordinates.json')
json = json.load(jsonFile)
folder = "C:/Users/coenc/PycharmProjects/Vision/assets"
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

trainFiles = []
y_train = []

mask = [[-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]]

for image in onlyfiles:
  trainFiles.append(image)

for coordinates in json:
  print(coordinates)

print("leng imageArray: ", len(trainFiles))

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()


# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

num_filters = 8
filter_size = 3
pool_size = 2

# Build the model.
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
#model.fit(
#  train_images,
#  to_categorical(train_labels),
#  epochs=1,
#  validation_data=(test_images, to_categorical(test_labels)),
#)