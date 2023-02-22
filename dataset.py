import numpy as np
import os
from PIL import Image
import json


def getData():
    images = []
    print("getting training data")
    counter = 0
    for filename in os.listdir('assets'):
        if filename.endswith('.jpg') or filename.endswith('.JPG'):
            img_path = os.path.join('assets', filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            images.append(img_array)
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
