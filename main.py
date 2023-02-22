import bunnyfinder
import dataset


def main():
    data = dataset
    train_data = data.getTrainData()
    print(train_data.shape)
    bunny = bunnyfinder
    bunny.bunnyfinder.evaluate(bunny)


if __name__ == '__main__':
    main()
