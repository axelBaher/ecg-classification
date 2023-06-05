from keras import Model, Sequential
from keras import layers
from keras.layers import *


class ModelLeNet5:
    def __init__(self):
        self.model_name = "LeNet-5"
        print(f"Start generating model {self.model_name} generating!")
        self.model = self.construct_model("categorical_crossentropy", "adam")
        print(f"Model {self.model_name} generated!")

    @staticmethod
    def construct_model(loss, optimizer):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(3, 3), activation="relu", input_shape=(128, 128, 1)))
        model.add(AveragePooling2D())
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
        model.add(AveragePooling2D())
        model.add(Flatten())
        model.add(Dense(units=120, activation="relu"))
        model.add(Dense(units=84, activation="relu"))
        model.add(Dense(units=3, activation="softmax"))
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return model


class ModelAlexNet:
    def __init__(self):
        self.model_name = "AlexNet"
        print(f"Start generating model {self.model_name} generating!")
        self.model = self.construct_model("categorical_crossentropy", "adam")
        print(f"Model {self.model_name} generated!")

    @staticmethod
    def construct_model(loss, optimizer):
        model = Sequential()
        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape=(128, 128, 1)))
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
        model.add(Dense(3, activation="softmax"))
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return model


class ModelVGGNetD:
    def __init__(self):
        self.model_name = "VGGNet-D"
        print(f"Start generating model {self.model_name} generating!")
        self.model = self.construct_model("categorical_crossentropy", "adam")
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

    def construct_model(self, loss, optimizer):
        model = Sequential()
        self.model = model

        self.model.add(InputLayer(input_shape=(128, 128, 1)))

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
        model.add(Dense(3, activation="softmax"))

        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return model


class ModelGoogLeNet:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model_name = "GoogLeNet"
        print(f"Start generating model {self.model_name} generating!")
        self.model = self.build("categorical_crossentropy", "adam")
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

    def build(self, loss, optimizer):
        input_layer = Input(shape=self.input_shape)

        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="valid", activation="relu")(input_layer)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = Conv2D(filters=192, kernel_size=(3, 3), padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

        x = self.inception_block(x, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)
        x = self.inception_block(x, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = self.inception_block(x, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)

        x1 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
        x1 = Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation="relu")(x1)
        x1 = Flatten()(x1)
        x1 = Dense(1024, activation="relu")(x1)
        x1 = Dropout(0.7)(x1)
        x1 = Dense(3, activation="softmax")(x1)

        x = self.inception_block(x, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64)
        x = self.inception_block(x, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64)
        x = self.inception_block(x, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64)

        x2 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
        x2 = Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation="relu")(x2)
        x2 = Flatten()(x2)
        x2 = Dense(1024, activation="relu")(x2)
        x2 = Dropout(0.7)(x2)
        x2 = Dense(3, activation="softmax")(x2)

        x = self.inception_block(x, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = self.inception_block(x, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)
        x = self.inception_block(x, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4=128)

        x = GlobalAveragePooling2D(name="GAPL")(x)
        x = Dropout(0.4)(x)
        x = Dense(3, activation="softmax")(x)

        model = Model(input_layer, [x, x1, x2], name="GoogLeNet")
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return model


class ModelResNet34:
    def __init__(self, input_shape=(128, 128, 1), num_classes=3):
        self.model_name = "ResNet-34"
        self.input_shape = input_shape
        self.num_classes = num_classes
        print(f"Start generating model {self.model_name} generating!")
        self.model = self.construct_model("categorical_crossentropy", "adam")
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

    def construct_model(self, loss, optimizer):
        input_tensor = Input(shape=self.input_shape)

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
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return model


"""
class ResBlock(Model):
    def __init__(self, filters, stride=1):
        super(ResBlock, self).__init__(name="ResBlock")
        self.flag = (stride != 1)
        self.conv_1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(stride, stride), padding="same")
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(filters=filters, kernel_size=(3, 3), padding="same")
        self.bn_2 = BatchNormalization()
        self.relu = ReLU()
        if self.flag:
            self.bn_3 = BatchNormalization()
            self.conv_3 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(stride, stride))

    def call(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.bn_1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.conv_2(x_1)
        x_1 = self.bn_2(x_1)
        if self.flag:
            x = self.conv_3(x)
            x = self.bn_3(x)
        x_1 = layers.add([x, x_1])
        x_1 = self.relu(x_1)
        return x_1


class ResNet34(Model):
    def __init__(self):
        super(ResNet34, self).__init__(name="ResNet34")
        self.conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.conv_2_1 = ResBlock(filters=64)
        self.conv_2_2 = ResBlock(filters=64)
        self.conv_2_3 = ResBlock(filters=64)

        self.conv_3_1 = ResBlock(filters=128, stride=2)
        self.conv_3_2 = ResBlock(filters=128)
        self.conv_3_3 = ResBlock(filters=128)
        self.conv_3_4 = ResBlock(filters=128)

        self.conv_4_1 = ResBlock(filters=256, stride=2)
        self.conv_4_2 = ResBlock(filters=256)
        self.conv_4_3 = ResBlock(filters=256)
        self.conv_4_4 = ResBlock(filters=256)
        self.conv_4_5 = ResBlock(filters=256)
        self.conv_4_6 = ResBlock(filters=256)

        self.conv_5_1 = ResBlock(filters=512, stride=2)
        self.conv_5_2 = ResBlock(filters=512)
        self.conv_5_3 = ResBlock(filters=512)

        self.pool = GlobalAveragePooling2D()
        self.fc_1 = Dense(512, activation="relu")
        self.dp_1 = Dropout(0.5)
        self.fc_2 = Dense(512, activation="relu")
        self.dp_2 = Dropout(0.5)
        self.fc_3 = Dense(64)

    def call(self, x):
        x = self.conv_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.conv_2_3(x)

        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.conv_3_4(x)

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.conv_4_4(x)
        x = self.conv_4_5(x)
        x = self.conv_4_6(x)

        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)

        x = self.pool(x)
        x = self.fc_1(x)
        x = self.dp_1(x)
        x = self.fc_2(x)
        x = self.dp_2(x)
        x = self.fc_3(x)
        return x
"""
