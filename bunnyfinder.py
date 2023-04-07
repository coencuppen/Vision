import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
from tensorflow_hub import KerasLayer
import dataset

IMAGE_SHAPE = (224, 224)


def train(epochs):
    # this function will train a model that will be used to calculate the change of a part of an image contains a bunny

    global IMAGE_SHAPE, model, bunniesArray, randomPicturesArray

    # get all the needed images for the training
    bunniesArray = dataset.getPictures('bunnies', IMAGE_SHAPE)
    # moreBunniesArray = dataset.getPictures('moreBunnies', IMAGE_SHAPE)
    randomPicturesArray = dataset.getPictures('randompictures', IMAGE_SHAPE)

    # getting all the needed labels and put them in a np.array()
    labels = []
    for i in range(804):
        labels.append(True)
    for i in range(804):
        labels.append(False)
    labels = np.array(labels)

    # putting all the images together in a np.array()
    data = []
    counter = 0
    for i in bunniesArray:
        rotatedImage = dataset.rotate(i)
        cuttedImage = dataset.randomCuts(i)

        data.append(i)
        data.append(rotatedImage)
        data.append(cuttedImage)

        # plt.imsave('checkpictures/'+counter.__str__()+'.jpg', i)
        # counter += 1
        # plt.imsave('checkpictures/'+counter.__str__()+'.jpg', rotatedImage)
        # counter += 1
        # plt.imsave('checkpictures/'+counter.__str__()+'.jpg', cuttedImage)
        # counter += 1

    # for i in moreBunniesArray:
    #    data.append(i)
    for i in randomPicturesArray:
        data.append(i)
        counter += 1
        # plt.imsave('checkpictures/' + counter.__str__()+'.jpg', i)
    data = np.array(data)
    # split the training and testing images
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=0.20, shuffle=True)
    #feature_extractor_model = "tf2-preview_mobilenet_v2_feature_vector_4"
    #pretrained_model_without_top_layer: KerasLayer = hub.KerasLayer(
    #    feature_extractor_model, input_shape=IMAGE_SHAPE + (1,), trainable=False)

    #model = tf.keras.Sequential([
    #   pretrained_model_without_top_layer,
    #   tf.keras.layers.Dense(1, name='output')
    #])

    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model.fit(X_train, y_train,
              validation_split=0.2,
              use_multiprocessing=True,
              callbacks=[early_stopping],
              epochs=epochs)

    testLoss, testAccuracy = model.evaluate(X_test, y_test, verbose=2)
    print("evaluation accuracy ", testAccuracy)


def predict(image):
    global IMAGE_SHAPE
    # this function takes a part of an image or an array[] of images,
    # and calculates the probability of it containing a bunny
    image = np.array(image)

    # if the parameter 'image' is an array[]
    if image.ndim == 3 or image.ndim == 1:
        arr = []
        for i in image:
            arr.append(cv2.resize(i, IMAGE_SHAPE))
        npArr = np.array(arr)

        # use the trained model to predict

        prediction = model.predict(npArr)
        for i in range(len(npArr)):
            # print(npArr[i])
            plt.imshow(npArr[i])
            plt.title(prediction[i].__str__())
            plt.show()
    # if the 'image' parameter is a single image
    else:
        print(image)
        print(image.shape)
        image = cv2.resize(image, IMAGE_SHAPE)
        print(image.shape)
        plt.imshow(image)
        image = image[None, :, :, :]
        prediction = model.predict(image)
        plt.title(prediction[0].__str__())
        plt.show()

    return prediction


def start(path, trainingEpochs, numberOfPictures=None, numberOfChecks=0, poolingScales=[255]):
    # Train the model
    train(trainingEpochs)

    # getting the pictures where the bunnies are hidden
    photos = dataset.getPictures(path, number=numberOfPictures)
    # photos = photos[22:23]
    # loop through every bunny picture
    for i in range(len(photos)):
        coordinatesArr = []
        bunnyCandidateArr = []
        predictionArr = []

        # loop through every pooling scale and number of checks given,
        # and calculate the prediction of every possible bunny
        # the numberOfChecks parameter is used to check the number of the returned highest values (blocks)
        # from the maxPooling() function

        for poolingScale in poolingScales:
            for candidate in range(numberOfChecks):
                # get the bunnyCandidates with the coordinates
                bunnyCandidate, coordinates = dataset.findBunny(photos[i],
                                                                poolingScale=poolingScale,
                                                                name=i,
                                                                candidate=candidate)


                bunnyCandidate = np.array(bunnyCandidate)
                coordinatesArr.append(coordinates)
                bunnyCandidateArr.append(bunnyCandidate)

            predictions = predict(bunnyCandidateArr)
            for prediction in predictions:
                predictionArr.append(prediction)

            bunnyCandidateArr = []

        # get the index of the bunny and coordinates that has the highest change of containing the bunny
        highestPredictionIndex = np.argmax(predictionArr)

        # draw the output image with the original picture, bounding box around the found bunny,
        # and the calculated prediction from the model
        dataset.drawAndSave(photos[i], coordinatesArr[highestPredictionIndex], 'output/', i,
                            text=predictionArr[highestPredictionIndex][0].__str__())
