import bunnyfinder


def main():
    bunnyfinder.start('assets', numberOfPictures=None, numberOfChecks=3,
                      poolingScales=[[128, 255], [64, 128], [255, 378]],
                      trainingEpochs=10)


if __name__ == '__main__':
    main()

