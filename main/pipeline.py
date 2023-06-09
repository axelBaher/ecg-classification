import models as m
import train
import itertools


def generate_models():
    models = dict({
        "LeNet5": m.ModelLeNet5(),
        # "AlexNet": m.ModelAlexNet(),
        # "GoogLeNet": m.ModelGoogLeNet(),
        # "ResNet34": m.ModelResNet34()
        # "VGGNetD": m.ModelVGGNetD()
    })
    return models


def main():
    epochs = list([1, 2, 3])
    batch_size = list([16, 32, 256, 512])
    validation_split = list([0.10, 0.15, 0.20])
    params = list([epochs, batch_size, validation_split])
    param_combs = list(itertools.product(*params))
    models = generate_models()
    for model in models.values():
        model.model.build(input_shape=(1, 128, 128, 1))
        for param_config in param_combs:
            print(f"\n\nModel: {model.model_name}\n"
                  f"Epochs: {param_config[0]}\n"
                  f"Batch size: {param_config[1]}\n"
                  f"Validation split: {param_config[2]}\n\n")
            train.train(model.model, model.model_name,
                        param_config[0], param_config[1], param_config[2],
                        pipeline=True)


if __name__ == "__main__":
    main()
