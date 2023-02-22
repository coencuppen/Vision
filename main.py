import bunnyfinder
import dataset

def main():
    data = dataset
    train_data = data.getTrainData()
    train_labels = data.getLabels()
    print(train_data.shape)
    print(train_labels.shape)

    bunny = bunnyfinder
    bunny.bunnyfinder.evaluate(bunny)


if __name__ == '__main__':
    main()
