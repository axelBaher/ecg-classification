import train
import inference
import itertools


def main():
    epochs = list([1, 2])
    batch_size = list([32, 256])
    validation_split = list([0.20])
    # epochs = list([1])
    # batch_size = list([256, 512])
    # validation_split = list([0.10])
    loss = [
        "categorical_crossentropy"
    ]
    optimizer = [
        "adam"
    ]
    params = list([epochs, batch_size, validation_split, loss, optimizer])
    param_combs = list(itertools.product(*params))
    models = list([
        "LeNet5",
        "AlexNet",
        "GoogLeNet",
        "ResNet34"
        "VGGNetD"
    ])
    train_values, train_labels = train.data_processing()
    test_values, test_labels = inference.data_processing()

    for i, model in enumerate(models):
        for j, param_config in enumerate(param_combs):
            index = (j + (len(param_combs) * i)) + 1
            print(f"\n\n{index}/{len(param_combs) * len(models)} pipeline iteration\n\n"
                  f"Model: {model}\n"
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
            train.train(model_config, train_values, train_labels, pipeline=True)
            inference.inference(model_config, test_values, test_labels, pipeline=True)


if __name__ == "__main__":
    main()
