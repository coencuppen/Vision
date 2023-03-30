import bunnyfinder
import dataset

def more():
    labels = dataset.getLabels()
    dataset.cutBunnies(labels)

def main():
    bunnyfinder.train()  # Train a model

    photos = dataset.getPictures('assets')
    for i in range(len(photos)):
        bunnyCandidate, coordinates = dataset.findBunny(photos[i], name=i)
        prediction = bunnyfinder.predict(bunnyCandidate)[0][0]
        print('prediction', prediction)

        dataset.drawAndSave(photos[i], coordinates, 'output/', i, text=prediction.__str__())


if __name__ == '__main__':
    main()
