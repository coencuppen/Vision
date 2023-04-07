import bunnyfinder
import dataset


def main():
    bunnyfinder.start('assets', numberOfPictures=None, numberOfChecks=4, poolingScales=[128],
                      trainingEpochs=6)


if __name__ == '__main__':
    main()
