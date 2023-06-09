from tensorflow import keras
import models as m
import dataloader as dl
import numpy as np
import os
import argparse
from time import gmtime, strftime


EPOCHS = 3
NUM_CLASSES = 3
BATCH_SIZE = 256
EXTENSION = "png"
VALIDATION_SPLIT = 0.15


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


def save_training_weights(model: keras.Sequential, model_name: str):
    cur_date_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    file_name = f"{cur_date_time}.h5"
    path = f"../weights/{model_name}"
    os.makedirs(path, exist_ok=True)
    model.save_weights(path + "\\" + file_name)
    print(f"Model weights were saved into directory:\n{path + '/' + file_name}")


def train(model: keras.Sequential, model_name: str):
    print("Getting data!")
    train_data = dl.DataLoader(data_name="train",
                               batch_size=BATCH_SIZE,
                               extension=EXTENSION)
    test_data = dl.DataLoader(data_name="test",
                              batch_size=BATCH_SIZE,
                              extension=EXTENSION)
    print("Data obtained!\n")
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
    train_log = model.fit(
        x=train_data_values, y=train_data_labels,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT)
    save_training_weights(model, model_name)
    # model_1 = m.ModelLeNet5()
    # model_1.model.load_weights("../weights/LeNet-5/2023-06-09_09-47-44.h5")
    # print("Weights loaded!")
    # model_1.model.build(input_shape=(1, 128, 128, 1))
    # print("Testing new model with loaded weights!")
    # log = model_1.model.evaluate(test_data_values, test_data_labels)
    eval_log = model.evaluate(test_data_values, test_data_labels)

    # print(f"Model {model_name} log:")
    # print(f"Loss function value:\n{eval_log[0]}")
    # print(f"Accuracy value:\n{eval_log[1]}\n")
    print("End training!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    # args = parse_args()
    model = m.ModelResNet34()
    print("Model building!")
    model.model.build(input_shape=(1, 128, 128, 1))
    print(f"Model {model.model_name} build!\n")
    print("Start training!")
    train(model.model, model.model_name)


if __name__ == "__main__":
    main()
