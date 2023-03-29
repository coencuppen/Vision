import numpy as np
import os
import scipy.ndimage
from PIL import Image
import json
from matplotlib import pyplot as plt, patches
from scipy import ndimage
from scipy.signal import convolve2d
import bunnyfinder
import cv2


def getPictures(directory, size=None, number=None):
    images = []
    counter = 1
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
            if counter == number:
                break
            counter += 1
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
    height, width = image.shape
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


def drawAndSave(image, path,  nmbr):
    name = nmbr.__str__() + ".jpg"
    cv2.imwrite(path.__str__() + name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # getAx(image, label)
    # plt.savefig("bunnies/" + name.__str__())


def draw(image, label):
    getAx(image, label)
    plt.show()


def findBunny(image, scale=1, poolingScale=255, name=None):
    global coordinateBunny



    imageGray = np.mean(image, axis=-1) / 255

    mask = [[-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1]]

    mask1 = [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]]

    newImage = cv2.resize(imageGray, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    newImage = scipy.ndimage.convolve(newImage, mask1)
    # maxImage = np.max(newImage)
    # binaryImage = newImage > 0.2 * maxImage

    mask2 = [[0, 0, 0, 0, 0, 0],
             [0, -1, 0, 0, 0, 0],
             [-1, 1, -1, 0, 0, 0],
             [-1, 1, 1, 0, 0, 0],
             [-1, 1, 1, -1, 0, 0],
             [-1, 1, 1, 1, -1, 0]]

    mask3 = [[0, -1, 0, 0, 0, 0, 0, -1, 0],
             [-1, 2, -1, 0, 0, 0, -1, 2, -1],
             [0, -1, 2, -1, 0, -1, 2, -1, 0],
             [0, 0, -1, 2, -1, 2, -1, 0, 0],
             [0, 0, 0, -1, 3, -1, 0, 0, 0],
             [0, 0, 0, -1, 2, -1, 0, 0, 0],
             [0, 0, 0, -1, 2, -1, 0, 0, 0],
             [0, 0, 0, -1, 2, -1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    newImage = scipy.ndimage.convolve(newImage, mask3)

    poolingImage = max_pooling(newImage, poolingScale)

    for y in range(len(poolingImage)):
        for x in range(len(poolingImage[1])):
            if poolingImage[y][x] == poolingImage.max():
                coordinateBunny = ((x) * poolingScale.__int__(),
                                   (y) * poolingScale.__int__(),
                                   (x + 1) * poolingScale.__int__(),
                                   (y + 1) * poolingScale.__int__())

    croppedImage = image[coordinateBunny[1]:coordinateBunny[3], coordinateBunny[0]:coordinateBunny[2]]
    #plt.imshow(image)
    #plt.axis('off')
    #plt.show()
    #plt.imshow(poolingImage)
    #plt.axis('off')
    #plt.show()
    #plt.imshow(croppedImage)
    #plt.axis('off')
    #plt.show()
    #drawAndSave(croppedImage, 'bunniesFound/', name)
    return croppedImage


def createMask(image):
    imageCanny = cv2.Canny(image, 100, 200)
    print(imageCanny.shape)
    counter = 0
    imageCanny = (imageCanny / 255)
    for y in range(len(imageCanny[0])):
        for x in range(len(imageCanny[1])):
            if imageCanny[x][y] > 0:
                counter += 1

    temp = counter / (imageCanny.shape[0] * imageCanny.shape[1])
    print(temp)
    for y in range(len(imageCanny[0])):
        for x in range(len(imageCanny[1])):
            if imageCanny[x][y] == 0.0:
                imageCanny[x][y] = -temp

    print(counter)
    plt.imshow(imageCanny)
    plt.show()
    return imageCanny
