from tensorflow import keras
import models as m
import dataloader as dl
import numpy as np
import os
import argparse
from time import gmtime, strftime
import json


EPOCHS = 5
NUM_CLASSES = 3
BATCH_SIZE = 256
EXTENSION = "png"
VALIDATION_SPLIT = 0.15


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def save_training_weights(model: keras.Sequential, model_name: str):
    cur_date_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    file_name = f"{cur_date_time}.h5"
    path = f"../weights/{model_name}"
    os.makedirs(path, exist_ok=True)
    model.save_weights(path + "\\" + file_name)
    print(f"Model weights were saved into directory:\n{path + '/' + file_name}")


def data_processing():
    print("Getting data!")
    train_data = dl.DataLoader(data_name="train",
                               batch_size=BATCH_SIZE,
                               extension=EXTENSION)
    test_data = dl.DataLoader(data_name="test",
                              batch_size=BATCH_SIZE,
                              extension=EXTENSION)
    train_values = train_data.get_data()
    test_values = test_data.get_data()
    print("Data obtained!\n")
    print("Processing data!")
    train_values = np.expand_dims(train_values, axis=3)
    test_values = np.expand_dims(test_values, axis=3)

    with open("../data/class-mapper.json") as f:
        class_mapper = json.load(f)
    train_labels = list()
    test_labels = list()
    for elem in train_data.data:
        train_labels.append(class_mapper[elem["label"]])
    for elem in test_data.data:
        test_labels.append(class_mapper[elem["label"]])
    train_labels = keras.utils.to_categorical(train_labels, NUM_CLASSES)
    test_labels = keras.utils.to_categorical(test_labels, NUM_CLASSES)
    print("Data processed!\n")
    return train_values, train_labels, test_values, test_labels


def train(model: keras.Sequential, model_name: str):
    train_values, train_labels, test_values, test_labels = data_processing()
    print(f"Model {model_name} training started!")
    train_log = model.fit(
        x=train_values, y=train_labels,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT)
    # save_training_weights(model, model_name)
    # model_1 = m.ModelLeNet5()
    # model_1.model.load_weights("../weights/LeNet-5/2023-06-09_09-47-44.h5")
    # print("Weights loaded!")
    # model_1.model.build(input_shape=(1, 128, 128, 1))
    # print("Testing new model with loaded weights!")
    # log = model_1.model.evaluate(test_values, test_labels)
    eval_log = model.evaluate(test_values, test_labels)
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
    model = m.ModelGoogLeNet()
    print("Model building!")
    model.model.build(input_shape=(1, 128, 128, 1))
    print(f"Model {model.model_name} build!\n")
    train(model.model, model.model_name)


if __name__ == "__main__":
    main()
