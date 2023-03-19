import numpy as np
import os

import scipy.ndimage
from PIL import Image
import json
from matplotlib import pyplot as plt, patches
import bunnyfinder
import cv2


def getPictures(directory, size=None):
    images = []
    print("getting", directory)
    for filename in os.listdir(directory):
        # print(filename)
        if filename.endswith('.jpg') or filename.endswith('.JPG'):
            img_path = os.path.join(directory, filename)
            if size:
                image = np.array(Image.open(img_path).resize(size))
            else:
                image = np.array(Image.open(img_path))

            images.append(image)
    return np.array(images)


def getLabels():
    labels = []
    print("getting labels")
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
        for name in data["pictures"]:
            labels.append([data["pictures"][name][0]["begin"], data["pictures"][name][0]["end"]])
    return np.array(labels)


def cutBunnies(labels):
    images = []
    counter = 0
    for filename in os.listdir('assets'):
        print(filename)
        if filename.endswith('.jpg') or filename.endswith('.JPG'):
            img_path = os.path.join('assets', filename)
            image = Image.open(img_path)
            img_cropped = image.crop((labels[counter][0][0], labels[counter][0][1],
                                      labels[counter][1][0], labels[counter][1][1]))
            drawAndSave(np.array(img_cropped), counter)
            counter += 1

    bunniesArray = np.array(images)
    print(bunniesArray.shape)


def max_pooling(image, n):
    height, width, color = image.shape
    newHeight, newWidth = height // n, width // n
    outputArray = np.zeros((newHeight, newWidth))

    for i in range(newHeight):
        for j in range(newWidth):
            row_start = i * n
            row_end = (i + 1) * n
            col_start = j * n
            col_end = (j + 1) * n
            block = image[row_start:row_end, col_start:col_end]
            outputArray[i, j] = np.max(block)

    return outputArray


def getAx(image, label):
    fig, ax = plt.subplots()
    ax.imshow(image)
    boundingBox = patches.Rectangle((label[0][0], label[0][1]), label[1][0] - label[0][0], label[1][1] - label[0][1],
                                    linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(boundingBox)


def drawAndSave(image, nmbr):
    name = nmbr.__str__() + ".jpg"
    cv2.imwrite('bunnies/' + name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # getAx(image, label)
    # plt.savefig("bunnies/" + name.__str__())


def draw(image, label):
    getAx(image, label)
    plt.show()


def findBunny(image):
    image = np.mean(image, axis=-1)/255

    earMask = [[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, -2, 0, 0, 0],
               [0, 0, -1, 4, -1, 0, 0],
               [0, 0, -1, 8, -1, 0, 0],
               [0, 0, -1, 4, -1, 0, 0],
               [0, 0, 0, -2, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]]
    newImage = scipy.ndimage.convolve(image, earMask)
    print(image)
    plt.imshow(newImage)
    plt.axis('off')
    plt.show()


photos = getPictures('assets')
findBunny(photos[0])
exit()
