from tensorflow import keras
import model as m
import dataloader as dl
import numpy as np

EPOCHS = 5
NUM_CLASSES = 8
BATCH_SIZE = 16
EXTENSION = "png"


def train(model: keras.Sequential):
    print("Model compiling!")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    print("Model compiled!\n")
    # (33 + 2238 + 1) * 2 = 4544
    print("Getting train, valid data!")
    train_data = dl.DataLoader(data_name="train",
                               batch_size=BATCH_SIZE,
                               extension=EXTENSION)
    valid_data = dl.DataLoader(data_name="valid",
                               batch_size=BATCH_SIZE,
                               extension=EXTENSION)
    print("Train, valid data obtained!\n")
    # print("Splitting data by labels!")
    # train_data_splitted = train_data.data_split()
    # valid_data_splitted = valid_data.data_split()
    # print("Data splitted by labels!\n")
    print("Getting train, valid data values!")
    train_data_values = train_data.get_data()
    valid_data_values = valid_data.get_data()
    print("Train, valid data values obtained!\n")
    train_data_labels = list()
    valid_data_labels = list()
    for elem in train_data.data:
        label = elem["label"]
        match label:
            case "N":
                label = 0
            case "A":
                label = 1
            case "V":
                label = 2
        train_data_labels.append(label)
    for elem in valid_data.data:
        label = elem["label"]
        match label:
            case "N":
                label = 0
            case "A":
                label = 1
            case "V":
                label = 2
        valid_data_labels.append(label)
    train_data_values = np.expand_dims(train_data_values, axis=3)
    valid_data_values = np.expand_dims(valid_data_values, axis=3)
    train_data_labels = keras.utils.to_categorical(train_data_labels, 64)
    valid_data_labels = keras.utils.to_categorical(valid_data_labels, 64)
    history = model.fit(
        x=train_data_values, y=train_data_labels,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=(valid_data_values, valid_data_labels))
    print(history)


def main():
    print("Start!\n")
    print("Model generating!")
    model = m.ResNet34()
    print("Model generated!\n")
    print("Model building!")
    model.build(input_shape=(1, 128, 128, 1))
    print("Model builded!\n")
    print("Entering train loop!")
    train(model)


if __name__ == "__main__":
    main()
