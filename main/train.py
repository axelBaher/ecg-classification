from tensorflow import keras
import models as m
import dataloader as dl
import numpy as np
import os
import argparse

EPOCHS = 1
NUM_CLASSES = 8
BATCH_SIZE = 16
EXTENSION = "png"
VALIDATION_SPLIT = 0.15

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def temp_label_split(data):
    labels = list()
    for elem in data:
        label = elem["label"]
        match label:
            case "N":
                label = 0
            case "A":
                label = 1
            case "V":
                label = 2
        labels.append(label)
    return labels


def save_training_weights():
    pass


def data_processing():
    pass


def train(model: keras.Sequential, model_name: str):
    print("Getting data!")
    train_data = dl.DataLoader(data_name="train",
                               batch_size=BATCH_SIZE,
                               extension=EXTENSION)
    test_data = dl.DataLoader(data_name="test",
                              batch_size=BATCH_SIZE,
                              extension=EXTENSION)
    print("Data obtained!\n")
    # print("Splitting data by labels!")
    # train_data_splitted = train_data.data_split()
    # test_data_splitted = test_data.data_split()
    # print("Data splitted by labels!\n")
    print("Getting data values!")
    train_data_values = train_data.get_data()
    test_data_values = test_data.get_data()
    print("Data values obtained!\n")
    train_data_labels = temp_label_split(train_data.data)
    test_data_labels = temp_label_split(test_data.data)
    train_data_values = np.expand_dims(train_data_values, axis=3)
    test_data_values = np.expand_dims(test_data_values, axis=3)
    train_data_labels = keras.utils.to_categorical(train_data_labels, NUM_CLASSES)
    test_data_labels = keras.utils.to_categorical(test_data_labels, NUM_CLASSES)

    print(f"Model {model_name} training started!")
    model.fit(
        x=train_data_values, y=train_data_labels,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT)

    log = model.evaluate(test_data_values, test_data_labels)

    print(f"Model {model_name} log:")
    print(f"Loss function value:\n{log[0]}")
    print(f"Accuracy value:\n{log[1]}\n")
    print("End training!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Start!\n")
    print("Model generating!")
    model = m.ModelLeNet5()
    print(f"Model {model.model_name} generated!\n")
    print("Model building!")
    model.model.build(input_shape=(1, 128, 128, 1))
    print(f"Model {model.model_name} build!\n")
    print("Start training!")
    train(model.model, model.model_name)


if __name__ == "__main__":
    main()
