import train
import itertools


def main():
    epochs = list([1, 2, 3])
    batch_size = list([16, 32, 256, 512])
    validation_split = list([0.10, 0.15, 0.20])
    loss = ["categorical_crossentropy"]
    optimizer = ["adam"]
    params = list([epochs, batch_size, validation_split, loss, optimizer])
    param_combs = list(itertools.product(*params))
    models = list([
        "LeNet5"
        # ,
        # "AlexNet",
        # "GoogLeNet",
        # "ResNet34",
        # "VGGNetD"
    ])

    for model in models:
        for param_config in param_combs:
            print(f"\n\nModel: {model}\n"
                  f"Epochs: {param_config[0]}\n"
                  f"Batch size: {param_config[1]}\n"
                  f"Validation split: {param_config[2]}\n"
                  f"Loss function: {param_config[3]}\n"
                  f"Optimizer: {param_config[4]}\n")
            model_config = {
                "name": model,
                "epochs": param_config[0],
                "batch-size": param_config[1],
                "validation-split": param_config[2],
                "loss": param_config[3],
                "optimizer": param_config[4]
            }
            train.train(model_config, pipeline=True)


if __name__ == "__main__":
    main()
