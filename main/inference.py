from tensorflow import keras
import models as m
import dataloader as dl
import numpy as np
import os
import argparse
import json
from keras.callbacks import CSVLogger
import pytz
from datetime import datetime

EXTENSION = "png"


def data_processing():
    print("Getting data!")
    test_data = dl.DataLoader(data_name="test",
                              extension=EXTENSION)
    test_values, test_labels = test_data.get_processed_data()
    return test_values, test_labels


def find_file(filename, search_path):
    # model_name, file_name = os.path.split(filename)
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            path = root + '/' + filename
            return path
    return None


def load_training_weights(model: keras.Sequential, model_config, cur_date_time, pipeline: bool = False):
    model_name = model_config["name"]
    epochs = model_config["epochs"]
    batch_size = model_config["batch-size"]
    validation_split = model_config["validation-split"]
    file_name = f"{epochs}-{batch_size}-{validation_split}.h5"
    # if pipeline:
    #     search_path = f"../weights/pipeline/{model_name}"
    # else:
    #     search_path = f"../weights/train/{model_name}"
    search_path = f"../weights/{model_name}"
    try:
        result = find_file(file_name, search_path)
    except FileNotFoundError:
        print("There is not such file with weights!")
        exit(-1)
    model.model.load_weights(result)
    print(f"Model weights were loaded from directory:\n{result}")
    return model


def get_log_path(model_config, cur_date_time, pipeline: bool = False):
    model_name, epochs, batch_size, validation_split = m.get_model_config_params(model_config)
    file_name = f"{epochs}-{batch_size}-{validation_split}.csv"
    if pipeline:
        path = f"../log/pipeline/test/{model_name}"
    else:
        path = f"../log/test/{model_name}"
    os.makedirs(path, exist_ok=True)
    full_path = path + '/' + file_name
    csv_logger = CSVLogger(full_path, append=True, separator='\n')
    print(f"Log will be saved into directory:\n{path + '/' + file_name}")
    return csv_logger


def inference(model_config, test_values, test_labels, cur_date_time, pipeline: bool = False):
    # model, model_name, epochs, batch_size, validation_split = m.get_model_config_params(model_config, pipeline)
    model_name = model_config["name"]
    model = m.get_model(model_config)
    print(f"Model {model_name} testing started!")

    csv_logger = get_log_path(model_config, cur_date_time, pipeline)
    model = load_training_weights(model, model_config, cur_date_time, pipeline)
    history = model.model.evaluate(test_values, test_labels, callbacks=[csv_logger])
    history.insert(0, "accuracy")
    history.insert(0, "loss")
    with open(csv_logger.filename, 'w') as f:
        f.writelines("%s\n" % line for line in history)
    print(f"Model {model_name} testing finished!")
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--b_size", type=int, required=True)
    parser.add_argument("--val_split", type=float, required=True)
    return parser.parse_args()


def main():
    console = 0
    if console:
        args = parse_args()
        model_config = {
            "name": args.name,
            "epochs": args.epoch,
            "batch-size": args.b_size,
            "validation-split": args.val_split,
            "loss": "categorical_crossentropy",
            "optimizer": "adam"
        }
    else:
        model_config = {
            "name": "LeNet5",
            "epochs": 1,
            "batch-size": 32,
            "validation-split": 0.2,
            "loss": "categorical_crossentropy",
            "optimizer": "adam"
        }
    timezone = pytz.timezone('Europe/Moscow')
    current_datetime = datetime.now(timezone)
    cur_date_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    test_values, test_labels = data_processing()
    inference(model_config, test_values, test_labels, cur_date_time)


if __name__ == "__main__":
    main()
