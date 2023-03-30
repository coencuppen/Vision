import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import cv2
from tensorflow_hub import KerasLayer

import dataset
from matplotlib import pyplot as plt, patches


def train():
    global model, bunniesArray, randomPicturesArray
    IMAGE_SHAPE = (224, 224)
    bunniesArray = dataset.getPictures('bunnies', IMAGE_SHAPE)
    randomPicturesArray = dataset.getPictures('randompictures', IMAGE_SHAPE)
    labels = []
    for i in range(268):
        labels.append(1)
    for i in range(529):
        labels.append(0)
    labels = np.array(labels)

    data = []
    for i in bunniesArray:
        data.append(i)
    for i in randomPicturesArray:
        data.append(i)

    data = np.array(data)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=True, shuffle=True)

    IMAGE_SHAPE = IMAGE_SHAPE + (3,)
    feature_extractor_model = "tf2-preview_mobilenet_v2_feature_vector_4"
    pretrained_model_without_top_layer: KerasLayer = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    model = tf.keras.Sequential([
        pretrained_model_without_top_layer,
        tf.keras.layers.Dense(1, name='output')
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, y_train,
              validation_split=0.2,
              use_multiprocessing=True,
              callbacks=[early_stopping],
              epochs=10)  # Epochs 8

    testLoss, testAccuracy = model.evaluate(X_test, y_test, verbose=2)
    print("evaluation accuracy ", testAccuracy)


def predict(image):
    image = np.array(image)
    if image.ndim == 4:
        arr = []
        for i in image:
            arr.append(cv2.resize(i, (224, 224)))
        npArr = np.array(arr)

        prediction = model.predict(npArr)

    else:
        image = cv2.resize(image, (224, 224))
        image = image[None, :, :, :]
        prediction = model.predict(image)

    return prediction

