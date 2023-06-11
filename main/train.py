from tensorflow import keras
import models as m
import dataloader as dl
import os
import argparse
from keras.callbacks import CSVLogger


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
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def train(model_config, train_values, train_labels, pipeline: bool = False):
    model = m.get_model(model_config)
    model_name = model_config["name"]
    print(f"Model {model_name} training started!")
    csv_logger = get_log_path(model_config, pipeline)
    model.model.fit(
        x=train_values, y=train_labels,
        epochs=model_config["epochs"],
        batch_size=model_config["batch-size"],
        validation_split=model_config["validation-split"],
        callbacks=[csv_logger])
    save_training_weights(model, model_config)
    print("End training!")


def main():
    args = parse_args()
    model_name = args.config
    model_config = m.get_model_config(model_name)
    train_values, train_labels = data_processing()
    train(model_config, train_values, train_labels)


if __name__ == "__main__":
    main()
