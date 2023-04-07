import numpy as np
import os
import scipy.ndimage
import scipy
from PIL import Image
import json
from matplotlib import pyplot as plt, patches
from skimage import transform
import random
from scipy.signal import convolve2d
import bunnyfinder
import cv2


def getPictures(directory, size=None, number=None):
    # this function reads all the .jpg images from a given folden, and returns a np.array with all the images
    images = []
    counter = 1
    print("getting", directory)
    for filename in os.listdir(directory):
        # print(filename)
        if filename.endswith('.jpg') or filename.endswith('.JPG'):
            img_path = os.path.join(directory, filename)
            if size:
                image = Image.open(img_path)
                longer_side = max(image.width, image.height)
                new_image = Image.new('RGB', (longer_side, longer_side), (0, 0, 0))
                x_offset = (longer_side - image.width) // 2
                y_offset = (longer_side - image.height) // 2
                new_image.paste(image, (x_offset, y_offset))
                new_image = new_image.convert('L')
                image = np.array(new_image.resize(size)) / 255
            else:
                image = Image.open(img_path)
                new_image = image.convert('L')
                image = np.array(new_image) / 255

            images.append(image)
            if counter == number:
                break
            counter += 1
    return np.array(images)


def getLabels():
    # this function reads all the coordinates from the .json file and puts them in a np.array()
    # this function is used to give the cutBunnies() function the coordinates of all the bunnies that are hiding
    labels = []
    print("getting labels")
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
        for name in data["pictures"]:
            labels.append([data["pictures"][name][0]["begin"], data["pictures"][name][0]["end"]])
    return np.array(labels)


def rotate(image):
    newImage = transform.rotate(image, random.randrange(-100, 100))
    return newImage


def randomCuts(image):
    randomValue = random.randrange(40, 80)
    newImage = image[randomValue: image.shape[0] - randomValue,
               randomValue: image.shape[0] - randomValue]
    newImage = np.array(Image.fromarray(newImage).resize(bunnyfinder.IMAGE_SHAPE))
    return newImage


def cutBunnies(labels):
    # this function cuts all the bunnies out of the pictures from the '/assets' folder
    # this is needed to give the model data to train on
    images = []
    counter = 0
    for filename in os.listdir('assets'):
        print(filename)
        if filename.endswith('.jpg') or filename.endswith('.JPG'):
            img_path = os.path.join('assets', filename)
            image = Image.open(img_path)
            img_cropped = image.crop((labels[counter][0][0] + 650, labels[counter][0][1]-250,
                                      labels[counter][1][0] + 650, labels[counter][1][1]-250))
            drawAndSave(np.array(img_cropped), name=counter + 1212, path='moreRandomPictures/')
            counter += 1

    bunniesArray = np.array(images)
    print(bunniesArray.shape)


def max_pooling(image, n):
    # this function takes the image and generates a down sampled max pooling array
    # the highest value of the array is where the bunny could be in
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
    # this function is used to draw the bounding box on the output image
    #
    fig, ax = plt.subplots()
    ax.imshow(image)
    if type(label) == tuple:
        boundingBox = patches.Rectangle((label[0], label[1]), label[2] - label[0], label[3] - label[1],
                                        linewidth=1, edgecolor='r', facecolor='none')

    else:
        boundingBox = patches.Rectangle((label[0][0], label[0][1]), label[1][0] - label[0][0],
                                        label[1][1] - label[0][1],
                                        linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(boundingBox)


def drawAndSave(image, label=None, path=None, name=None, text=""):
    name = name.__str__() + ".jpg"
    # cv2.imwrite(path.__str__() + name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # return
    getAx(image, label)
    plt.axis('off')
    plt.text(label[0], (label[1] - 5), s=text, color='red')
    plt.savefig(path + name.__str__())
    plt.close()


def draw(image, label):
    getAx(image, label)
    plt.show()


def findBunny(image, scale=1, poolingScale=255, name=None, candidate=0):
    global coordinateBunny

    maskEdge = [[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]]

    newImage = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    newImage = scipy.ndimage.convolve(newImage, maskEdge)

    maskBunny = [[0, -1, 0, 0, 0, 0, 0, -1, 0],
                 [-1, 2, -1, 0, 0, 0, -1, 2, -1],
                 [0, -1, 2, -1, 0, -1, 2, -1, 0],
                 [0, 0, -1, 2, -1, 2, -1, 0, 0],
                 [0, 0, 0, -1, 3, -1, 0, 0, 0],
                 [0, 0, 0, -1, 2, -1, 0, 0, 0],
                 [0, 0, 0, -1, 2, -1, 0, 0, 0],
                 [0, 0, 0, -1, 2, -1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    newImage = scipy.ndimage.convolve(newImage, maskBunny)

    poolingImage = max_pooling(newImage, poolingScale)

    candidates = []
    for i in poolingImage:
        for j in i:
            candidates.append(j)

    candidates.sort()
    candidate = candidates[-1 - candidate]

    for y in range(len(poolingImage)):
        for x in range(len(poolingImage[1])):
            if poolingImage[y][x] == candidate:
                coordinateBunny = (x * poolingScale.__int__(),
                                   y * poolingScale.__int__(),
                                   (x + 1) * poolingScale.__int__(),
                                   (y + 1) * poolingScale.__int__() + poolingScale)

    croppedImage = image[coordinateBunny[1]:coordinateBunny[3], coordinateBunny[0]:coordinateBunny[2]]

    imageArr = np.array(croppedImage)
    width = imageArr.shape[1]
    height = imageArr.shape[0]
    longer_side = max(imageArr.shape)
    newImage = np.zeros((longer_side, longer_side))

    x_offset = (longer_side - width) // 2
    y_offset = (longer_side - height) // 2

    for h in range(len(newImage)):
        for w in range(len(newImage[0])):
            if w >= width + x_offset:
                break
            if w > x_offset:
                newImage[h][w] = imageArr[h][w - x_offset]

    #plt.imshow(newImage)
    #plt.show()
    #plt.imshow(poolingImage)
    #plt.axis('off')
    #plt.show()
    #plt.imshow(croppedImage)
    #plt.axis('off')
    #plt.show()
    # drawAndSave(croppedImage, 'bunniesFound/', name)
    return newImage, coordinateBunny


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
