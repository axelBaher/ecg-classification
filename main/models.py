import keras
from keras import Model, Sequential
from keras.layers import *
import os
import json
import numpy as np


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
            model = ModelLeNet5(input_shape=(128, 128, 1), num_classes=len(class_mapper))
        case "AlexNet":
            model = ModelAlexNet(input_shape=(128, 128, 1), num_classes=len(class_mapper))
        case "VGGNetD":
            model = ModelVGGNetD(input_shape=(128, 128, 1), num_classes=len(class_mapper))
        case "GoogLeNet":
            model = ModelGoogLeNet(input_shape=(128, 128, 1), num_classes=len(class_mapper))
        case "ResNet34":
            model = ModelResNet34(input_shape=(128, 128, 1), num_classes=len(class_mapper))
    input_shape = np.expand_dims(model.input_data_shape, axis=0)
    model.model.build(input_shape)
    model.model.compile(loss=model_config["loss"],
                        optimizer=model_config["optimizer"],
                        metrics=["accuracy"])
    return model


def get_model_config_params(model_config, pipeline: bool = False):
    # if not pipeline:
    #     model = get_model(model_config)
    #     model_name = model_config["name"]
    #     epochs = model_config["epochs"]
    #     batch_size = model_config["batch-size"]
    #     validation_split = model_config["validation-split"]
    # else:
    model_name = model_config["name"]
    epochs = model_config["epochs"]
    batch_size = model_config["batch-size"]
    validation_split = model_config["validation-split"]
    return model_name, epochs, batch_size, validation_split


class ModelLeNet5:
    def __init__(self, input_shape, num_classes):
        self.model_name = "LeNet-5"
        self.num_classes = num_classes
        self.input_data_shape = input_shape
        print(f"Start generating model {self.model_name}!")
        self.model = self.construct_model()
        print(f"Model {self.model_name} generated!")

    def construct_model(self):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(3, 3), activation="relu", input_shape=self.input_data_shape))
        model.add(AveragePooling2D())
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
        model.add(AveragePooling2D())
        model.add(Flatten())
        model.add(Dense(units=120, activation="relu"))
        model.add(Dense(units=84, activation="relu"))
        model.add(Dense(units=self.num_classes, activation="softmax"))
        return model


class ModelAlexNet:
    def __init__(self, input_shape=(128, 128, 1), num_classes=3):
        self.model_name = "AlexNet"
        self.num_classes = num_classes
        self.input_data_shape = input_shape
        print(f"Start generating model {self.model_name}!")
        self.model = self.construct_model()
        print(f"Model {self.model_name} generated!")

    def construct_model(self):
        model = Sequential()
        model.add(
            Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape=self.input_data_shape))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(384, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(384, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation="softmax"))
        return model


class ModelVGGNetD:
    def __init__(self, input_shape=(128, 128, 1), num_classes=3):
        self.model_name = "VGGNet-D"
        self.num_classes = num_classes
        self.input_data_shape = input_shape
        print(f"Start generating model {self.model_name}!")
        self.model = self.construct_model()
        print(f"Model {self.model_name} generated!")

    def construct_block(self, layers_num: int,
                        filters: int, kernel_size: tuple,
                        padding: str, activation_function: str,
                        pooling_type: str, pool_size: tuple, strides: tuple):
        for _ in range(layers_num):
            self.model.add(Conv2D(filters=filters, kernel_size=kernel_size,
                                  padding=padding, activation=activation_function))
        match pooling_type:
            case "max":
                self.model.add(MaxPooling2D(pool_size=pool_size,
                                            strides=strides))
            case "avg":
                self.model.add(AveragePooling2D(pool_size=pool_size,
                                                strides=strides))

    def construct_model(self):
        model = Sequential()
        self.model = model

        self.model.add(InputLayer(input_shape=self.input_data_shape))

        self.construct_block(layers_num=2,
                             filters=64, kernel_size=(3, 3),
                             padding="same", activation_function="relu",
                             pooling_type="max", pool_size=(2, 2), strides=(2, 2))

        self.construct_block(layers_num=3,
                             filters=128, kernel_size=(3, 3),
                             padding="same", activation_function="relu",
                             pooling_type="max", pool_size=(2, 2), strides=(2, 2))

        self.construct_block(layers_num=3,
                             filters=256, kernel_size=(3, 3),
                             padding="same", activation_function="relu",
                             pooling_type="max", pool_size=(2, 2), strides=(2, 2))

        self.construct_block(layers_num=3,
                             filters=512, kernel_size=(3, 3),
                             padding="same", activation_function="relu",
                             pooling_type="max", pool_size=(2, 2), strides=(2, 2))

        model.add(Flatten())
        model.add(Dense(4096, activation="relu"))
        model.add(Dense(4096, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))
        return model


class ModelGoogLeNet:
    def __init__(self, input_shape=(128, 128, 1), num_classes=3):
        self.model_name = "GoogLeNet"
        self.input_data_shape = input_shape
        self.num_classes = num_classes
        print(f"Start generating model {self.model_name}!")
        self.model = self.construct_model()
        print(f"Model {self.model_name} generated!")

    @staticmethod
    def inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
        path1 = Conv2D(filters=f1, kernel_size=(1, 1), padding="same", activation="relu")(input_layer)

        path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding="same", activation="relu")(input_layer)
        path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), padding="same", activation="relu")(path2)

        path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding="same", activation="relu")(input_layer)
        path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), padding="same", activation="relu")(path3)

        path4 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(input_layer)
        path4 = Conv2D(filters=f4, kernel_size=(1, 1), padding="same", activation="relu")(path4)

        output_layer = concatenate([path1, path2, path3, path4], axis=-1)

        return output_layer

    def construct_block(self, x):
        _x = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
        _x = Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation="relu")(_x)
        _x = Flatten()(_x)
        _x = Dense(1024, activation="relu")(_x)
        _x = Dropout(0.7)(_x)
        _x = Dense(self.num_classes, activation="softmax")(_x)
        return _x

    def construct_model(self):
        input_layer = Input(shape=self.input_data_shape)

        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="valid", activation="relu")(input_layer)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = Conv2D(filters=192, kernel_size=(3, 3), padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

        x = self.inception_block(x, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)
        x = self.inception_block(x, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = self.inception_block(x, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)

        # x1 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
        # x1 = Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation="relu")(x1)
        # x1 = Flatten()(x1)
        # x1 = Dense(1024, activation="relu")(x1)
        # x1 = Dropout(0.7)(x1)
        # x1 = Dense(self.num_classes, activation="softmax")(x1)

        x1 = self.construct_block(x)

        x = self.inception_block(x, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64)
        x = self.inception_block(x, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64)
        x = self.inception_block(x, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64)

        x2 = self.construct_block(x)

        # x2 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
        # x2 = Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation="relu")(x2)
        # x2 = Flatten()(x2)
        # x2 = Dense(1024, activation="relu")(x2)
        # x2 = Dropout(0.7)(x2)
        # x2 = Dense(self.num_classes, activation="softmax")(x2)

        x = self.inception_block(x, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = self.inception_block(x, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)
        x = self.inception_block(x, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4=128)

        x = GlobalAveragePooling2D(name="GAPL")(x)
        x = Dropout(0.4)(x)
        x = Dense(self.num_classes, activation="softmax")(x)

        model = Model(input_layer, [x, x1, x2], name="GoogLeNet")
        return model


class ModelResNet34:
    def __init__(self, input_shape=(128, 128, 1), num_classes=3):
        self.model_name = "ResNet-34"
        self.input_data_shape = input_shape
        self.num_classes = num_classes
        print(f"Start generating model {self.model_name}!")
        self.model = self.construct_model()
        print(f"Model {self.model_name} generated!")

    @staticmethod
    def residual_block(input_tensor, filters, strides=(1, 1)):
        x = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding="same")(input_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)

        if strides != (1, 1) or input_tensor.shape[-1] != filters:
            input_tensor = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding="same")(input_tensor)
            input_tensor = BatchNormalization()(input_tensor)

        x = Add()([x, input_tensor])
        x = Activation("relu")(x)
        return x

    def construct_model(self):
        input_tensor = Input(shape=self.input_data_shape)

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same")(input_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        x = self.residual_block(x, filters=64)
        x = self.residual_block(x, filters=64)
        x = self.residual_block(x, filters=64)

        x = self.residual_block(x, filters=128, strides=(2, 2))
        x = self.residual_block(x, filters=128)
        x = self.residual_block(x, filters=128)
        x = self.residual_block(x, filters=128)

        x = self.residual_block(x, filters=256, strides=(2, 2))
        x = self.residual_block(x, filters=256)
        x = self.residual_block(x, filters=256)
        x = self.residual_block(x, filters=256)
        x = self.residual_block(x, filters=256)
        x = self.residual_block(x, filters=256)

        x = self.residual_block(x, filters=512, strides=(2, 2))
        x = self.residual_block(x, filters=512)
        x = self.residual_block(x, filters=512)

        x = AveragePooling2D(pool_size=(7, 7), padding="same")(x)
        x = Flatten()(x)
        x = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=input_tensor, outputs=x)
        return model
