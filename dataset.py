import numpy as np
import os
from PIL import Image


def getTrainData():
    images = []
    print("getting training data")
    for filename in os.listdir('assets'):
        print(filename)
        if filename.endswith('.jpg'):
            img_path = os.path.join('assets', filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            images.append(img_array)
    return np.array(images)


def getLabels():
    return 1