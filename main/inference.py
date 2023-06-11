from tensorflow import keras
import models as m
import dataloader as dl
import os
import argparse
from keras.callbacks import CSVLogger


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


def load_training_weights(model: keras.Sequential, model_config):
    model_name = model_config["name"]
    epochs = model_config["epochs"]
    batch_size = model_config["batch-size"]
    validation_split = model_config["validation-split"]
    file_name = f"{epochs}-{batch_size}-{validation_split}.h5"
    search_path = f"../weights/{model_name}"
    try:
        result = find_file(file_name, search_path)
    except FileNotFoundError:
        print("There is not such file with weights!")
        exit(-1)
    model.model.load_weights(result)
    print(f"Model weights were loaded from directory:\n{result}")
    return model


def get_log_path(model_config, pipeline: bool = False):
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


def inference(model_config, test_values, test_labels, pipeline):
    model_name = model_config["name"]
    model = m.get_model(model_config)
    print(f"Model {model_name} testing started!")

    csv_logger = get_log_path(model_config, pipeline)
    model = load_training_weights(model, model_config)
    history = model.model.evaluate(test_values, test_labels, callbacks=[csv_logger])
    history.insert(0, "accuracy")
    history.insert(0, "loss")
    with open(csv_logger.filename, 'w') as f:
        f.writelines("%s\n" % line for line in history)
    print(f"Model {model_name} testing finished!")
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str,
                        help="Model name",
                        required=True)
    parser.add_argument("-epoch", type=int,
                        help="Number of epochs were trained",
                        required=True)
    parser.add_argument("-b_size", type=int,
                        help="Used batch size while training",
                        required=True)
    parser.add_argument("-val_split", type=float,
                        help="Use validation split while training",
                        required=True)
    parser.add_argument("-loss", type=str,
                        help="Used loss function (categorical_crossentropy - default)",
                        required=False)
    parser.add_argument("-opt", type=str,
                        help="Used optimizer (adam - default)",
                        required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    model_config = {
        "name": args.name,
        "epochs": args.epoch,
        "batch-size": args.b_size,
        "validation-split": args.val_split,
        "loss": "categorical_crossentropy" if args.loss is None else args.loss,
        "optimizer": "adam" if args.opt is None else args.opt,
    }
    test_values, test_labels = data_processing()
    inference(model_config, test_values, test_labels, pipeline=False)


if __name__ == "__main__":
    main()
