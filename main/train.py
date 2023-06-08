from tensorflow import keras
import models as m
import dataloader as dl
import numpy as np
import os
import argparse
import json


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def get_model_config(model_name):
    model_config_path_abs = os.path.abspath(model_name).split("\\")
    model_config_path = "\\".join(model_config_path_abs[:-1]) + "\\" + \
                        "\\".join(["config", "training", model_name]) + ".json"
    try:
        with open(model_config_path) as p:
            model_config = json.load(p)
    except FileNotFoundError:
        print("Incorrect model name!")
        exit(-1)
    return model_config


def get_model(model_config):
    model_name = model_config["name"]
    model = None
    match model_name:
        case "LeNet5":
            model = m.ModelLeNet5()
        case "AlexNet":
            model = m.ModelAlexNet()
        case "VGGNetD":
            model = m.ModelVGGNetD()
        case "GoogLeNet":
            model = m.ModelGoogLeNet()
        case "ResNet34":
            model = m.ModelResNet34()
    return model


def train(model_config):
    model = get_model(model_config)
    model_name = model_config["name"]
    epochs = model_config["epochs"]
    batch_size = model_config["batch-size"]
    loss = model_config["loss"]
    optimizer = model_config["optimizer"]

    print("Getting data!")
    train_data = dl.DataLoader(data_name="train",
                               batch_size=batch_size,
                               extension=EXTENSION)
    test_data = dl.DataLoader(data_name="test",
                              batch_size=batch_size,
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

    print("Model building!")
    model.model.build(input_shape=(1, 128, 128, 1))
    print(f"Model {model.model_name} build!\n")

    print("Model compiling!")
    model.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    print(f"Model {model_name} training started!")
    model.model.fit(
        x=train_data_values, y=train_data_labels,
        batch_size=batch_size, epochs=epochs,
        validation_split=VALIDATION_SPLIT)

    log = model.model.evaluate(test_data_values, test_data_labels)

    print(f"Model {model_name} log:")
    print(f"Loss function value:\n{log[0]}")
    print(f"Accuracy value:\n{log[1]}\n")
    print("End training!")


def main():
    args = parse_args()
    model_name = args.config
    model_config = get_model_config(model_name)
    train(model_config)
    pass


if __name__ == "__main__":
    main()
