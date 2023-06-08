from tensorflow import keras
import models as m
import dataloader as dl
import numpy as np
import os
import argparse
from time import gmtime, strftime
import json
from keras.callbacks import CSVLogger


EPOCHS = 1
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.15
EXTENSION = "png"


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def save_training_weights(model: keras.Sequential, model_name: str, pipeline: bool = False):
    cur_date_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    file_name = f"{cur_date_time}.h5"
    if pipeline:
        path = f"../weights/pipeline/train/{model_name}"
    else:
        path = f"../weights/train/{model_name}"

    os.makedirs(path, exist_ok=True)
    model.save_weights(path + "\\" + file_name)
    print(f"Model weights were saved into directory:\n{path + '/' + file_name}")


def get_log_path(model_name: str, epochs, batch_size, validation_split, pipeline: bool = False):
    file_name = f"{epochs}-{batch_size}-{validation_split}.csv"
    if pipeline:
        path = f"../log/pipeline/train/{model_name}"
    else:
        path = f"../log/train/{model_name}"
    os.makedirs(path, exist_ok=True)
    csv_logger = CSVLogger(path + "\\" + file_name, append=True, separator='\n')
    print(f"Log will be saved into directory:\n{path + '/' + file_name}")
    return csv_logger


def data_processing(model_config):
    print("Getting data!")
    batch_size = model_config["batch-size"]
    train_data = dl.DataLoader(data_name="train",
                               batch_size=batch_size,
                               extension=EXTENSION)
    test_data = dl.DataLoader(data_name="test",
                              batch_size=batch_size,
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
    train_labels = keras.utils.to_categorical(train_labels, len(class_mapper))
    test_labels = keras.utils.to_categorical(test_labels, len(class_mapper))
    print("Data processed!\n")
    return train_values, train_labels, test_values, test_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def get_model_config(model_name):
    model_config_path_abs = os.path.abspath(f"../{model_name}").split("\\")
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
    with open("../data/class-mapper.json") as f:
        class_mapper = json.load(f)
    model_name = model_config["name"]
    model = None
    match model_name:
        case "LeNet5":
            model = m.ModelLeNet5(input_shape=(128, 128, 1), num_classes=len(class_mapper))
        case "AlexNet":
            model = m.ModelAlexNet()
        case "VGGNetD":
            model = m.ModelVGGNetD()
        case "GoogLeNet":
            model = m.ModelGoogLeNet()
        case "ResNet34":
            model = m.ModelResNet34()
    input_shape = np.expand_dims(model.input_data_shape, axis=0)
    model.model.build(input_shape)
    model.model.compile(loss=model_config["loss"],
                        optimizer=model_config["optimizer"],
                        metrics=["accuracy"])
    return model


def train(model_config, pipeline: bool = False):
    if not pipeline:
        model = get_model(model_config)
        model_name = model_config["name"]
        epochs = model_config["epochs"]
        batch_size = model_config["batch-size"]
        validation_split = model_config["validation-split"]
        loss = model_config["loss"]
        optimizer = model_config["optimizer"]
    else:
        model = get_model(model_config)
        model_name = model_config["name"]
        epochs = model_config["epochs"]
        batch_size = model_config["batch-size"]
        validation_split = model_config["validation-split"]

    train_values, train_labels, test_values, test_labels = data_processing(model_config)

    # print("Model building!")
    # model.model.build(input_shape=(1, 128, 128, 1))
    # print(f"Model {model_name} build!\n")

    print(f"Model {model_name} training started!")
    csv_logger = get_log_path(model_name, epochs, batch_size, validation_split, pipeline)
    model.model.fit(
        x=train_values, y=train_labels,
        batch_size=batch_size, epochs=epochs,
        validation_split=validation_split,
        callbacks=[csv_logger])
    save_training_weights(model, model_name, pipeline)
    # eval_log = model.model.evaluate(test_values, test_labels)
    print("End training!")


def main():
    console = 0
    if console:
        args = parse_args()
        model_name = args.config
    else:
        model_name = "LeNet5"
    model_config = get_model_config(model_name)
    train(model_config)


if __name__ == "__main__":
    main()
