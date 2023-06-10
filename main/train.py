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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def save_training_weights(model: keras.Sequential, model_config):
    model_name = model_config["name"]
    epochs = model_config["epochs"]
    batch_size = model_config["batch-size"]
    validation_split = model_config["validation-split"]
    file_name = f"{epochs}-{batch_size}-{validation_split}.h5"
    path = f"../weights/{model_name}"

    os.makedirs(path, exist_ok=True)
    model.model.save_weights(path + "\\" + file_name)
    print(f"Model weights were saved into directory:\n{path + '/' + file_name}")


def get_log_path(model_config, pipeline: bool = False):
    model_name, epochs, batch_size, validation_split = m.get_model_config_params(model_config)
    file_name = f"{epochs}-{batch_size}-{validation_split}.csv"
    if pipeline:
        path = f"../log/pipeline/train/{model_name}"
    else:
        path = f"../log/train/{model_name}"
    os.makedirs(path, exist_ok=True)
    csv_logger = CSVLogger(path + "\\" + file_name, append=True, separator='\n')
    print(f"Log will be saved into directory:\n{path + '/' + file_name}")
    return csv_logger


def data_processing():
    print("Getting data!")
    train_data = dl.DataLoader(data_name="train",
                               extension=EXTENSION)
    train_values, train_labels = train_data.get_processed_data()
    return train_values, train_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def train(model_config, train_values, train_labels, cur_date_time, pipeline: bool = False):
    # if not pipeline:
    #     model = get_model(model_config)
    #     model_name = model_config["name"]
    #     epochs = model_config["epochs"]
    #     batch_size = model_config["batch-size"]
    #     validation_split = model_config["validation-split"]
    # else:
    #     model = get_model(model_config)
    #     model_name = model_config["name"]
    #     epochs = model_config["epochs"]
    #     batch_size = model_config["batch-size"]
    #     validation_split = model_config["validation-split"]
    # model_name, epochs, batch_size, validation_split = m.get_model_config_params(model_config)
    model = m.get_model(model_config)
    model_name = model_config["name"]
    print(f"Model {model_name} training started!")
    csv_logger = get_log_path(model_config, cur_date_time)
    model.model.fit(
        x=train_values, y=train_labels,
        epochs=model_config["epochs"],
        batch_size=model_config["batch-size"],
        validation_split=model_config["validation-split"],
        callbacks=[csv_logger])
    save_training_weights(model, model_config)
    print("End training!")


def main():
    console = 1
    if console:
        args = parse_args()
        model_name = args.config
    else:
        model_name = "LeNet5"
    timezone = pytz.timezone('Europe/Moscow')
    current_datetime = datetime.now(timezone)
    cur_date_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    model_config = m.get_model_config(model_name)
    train_values, train_labels = data_processing()
    train(model_config, train_values, train_labels, cur_date_time)


if __name__ == "__main__":
    main()
