import numpy as np

import bunnyfinder
import dataset
import cv2


def main():
    bunnyfinder.train()  # Train a model
    #bunnyfinder.predict(bunnyfinder.bunniesArray[0:3])
    #bunnyfinder.predict(bunnyfinder.randomPicturesArray[0:3])
    photos = dataset.getPictures('assets')
    bunnies = []
    for i in range(len(photos)):
        bunnies.append(dataset.findBunny(photos[i], name=i))

    bunnyfinder.predict(bunnies)

if __name__ == '__main__':
    main()
