import train
import inference
import itertools
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline running")
    parser.add_argument("--config", help="Write \"pipeline\" to load pipeline config", required=True)
    return parser.parse_args()


def get_config(config_name):
    config_path = "../" + '/'.join(["config", f"{config_name}.json"])
    try:
        with open(config_path) as p:
            config = json.load(p)
    except FileNotFoundError:
        print("Cant get config for pipeline!")
        exit(-1)
    return config


def main():
    args = parse_args()
    config = args.config
    config = get_config(config)

    params = list([
        config["models"],
        config["epochs"],
        config["batch-size"],
        config["validation-split"],
        config["loss"],
        config["optimizer"]
    ])
    param_combs = list(itertools.product(*params))
    train_values, train_labels = train.data_processing()
    test_values, test_labels = inference.data_processing()
    for i, param_config in enumerate(param_combs):
        print(f"\n\n{i + 1}/{len(param_combs)} pipeline iteration\n\n"
              f"Model: {param_config[0]}\n"
              f"Epochs: {param_config[1]}\n"
              f"Batch size: {param_config[2]}\n"
              f"Validation split: {param_config[3]}\n"
              f"Loss function: {param_config[4]}\n"
              f"Optimizer: {param_config[5]}\n")
        model_config = {
            "name": param_config[0],
            "epochs": param_config[1],
            "batch-size": param_config[2],
            "validation-split": param_config[3],
            "loss": param_config[4],
            "optimizer": param_config[5]
        }
        train.train(model_config, train_values, train_labels, pipeline=True)
        inference.inference(model_config, test_values, test_labels, pipeline=True)


if __name__ == "__main__":
    main()
