# import train
import os
from dataloader import DataLoader
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
from keras import Model
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, Dense, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Input, concatenate, \
    GlobalAveragePooling2D, BatchNormalization, Activation, Add
from visualkeras import layered_view, graph_view, layered
import pydot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ModelLeNet5:
    def __init__(self):
        self.model_name = "LeNet-5"
        print(f"Start generating model {self.model_name} generating!")
        self.model = self.construct_model("categorical_crossentropy", "adam")
        print(f"Model {self.model_name} generated!")

    @staticmethod
    def construct_model(loss, optimizer):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
        model.add(AveragePooling2D())
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(AveragePooling2D())
        model.add(Flatten())
        model.add(Dense(units=120, activation='relu'))
        model.add(Dense(units=84, activation='relu'))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

    def save_model_figure(self):
        layer_node = 0
        graph = pydot.Dot(graph_type='digraph')

        input_node = pydot.Node("INPUT", shape="box")

        graph.add_node(input_node)

        prev_node = input_node
        i = 1
        for layer in self.model.layers[:-1]:
            layer_name = layer.__class__.__name__
            match layer_name:
                case "Conv2D":
                    layer_node = pydot.Node(f"{layer_name}_C{i}", shape="box",
                                            label=f"{layer_name}_C{i}\n{layer.input_shape}\n")
                    i += 1
                case "Dense":
                    layer_node = pydot.Node(f"{layer_name}_F{i}", shape="box")
                    i += 1
                case "AveragePooling2D":
                    layer_node = pydot.Node(f"{layer_name}_S{i}", shape="box")
                    i += 1
                case "Flatten":
                    layer_node = pydot.Node(f"{layer_name}", shape="box")

            graph.add_node(layer_node)
            graph.add_edge(pydot.Edge(prev_node, layer_node))
            prev_node = layer_node

        output_node = pydot.Node("OUTPUT", shape="box")
        graph.add_node(output_node)
        graph.add_edge(pydot.Edge(prev_node, output_node))

        image = graph.create(format='png')

        filename = f"model_figures/{self.model_name}.png"
        with open(filename, 'wb') as f:
            f.write(image)

        print(f"Diagram saved into the file: {filename}")


class ModelAlexNet:
    def __init__(self):
        self.model_name = "AlexNet"
        print(f"Start generating model {self.model_name} generating!")
        self.model = self.construct_model("categorical_crossentropy", "adam")
        print(f"Model {self.model_name} generated!")

    @staticmethod
    def construct_model(loss, optimizer):
        model = Sequential()
        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
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

        self.model.add(Input(input_shape=(224, 224, 3)))

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
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1000, activation='softmax'))

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model


# class ModelGoogLeNet:
#     def __init__(self):
#         self.model_name = "GoogLeNet"
#         print(f"Start generating model {self.model_name} generating!")
#         self.model = self.construct_model("categorical_crossentropy", "adam")
#         print(f"Model {self.model_name} generated!")
#
#     @staticmethod
#     def construct_model(loss, optimizer):
#         model = Sequential()
#
#         model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#         return model


class ModelGoogLeNet:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model_name = "GoogLeNet"
        print(f"Start generating model {self.model_name} generating!")
        self.model = self.build("categorical_crossentropy", "adam")
        print(f"Model {self.model_name} generated!")

    @staticmethod
    def inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
        path1 = Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

        path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
        path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), padding='same', activation='relu')(path2)

        path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
        path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), padding='same', activation='relu')(path3)

        path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
        path4 = Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(path4)

        output_layer = concatenate([path1, path2, path3, path4], axis=-1)

        return output_layer

    def build(self, loss, optimizer):
        input_layer = Input(shape=self.input_shape)

        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')(input_layer)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

        x = self.inception_block(x, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)
        x = self.inception_block(x, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = self.inception_block(x, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)

        x1 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
        x1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(x1)
        x1 = Flatten()(x1)
        x1 = Dense(1024, activation='relu')(x1)
        x1 = Dropout(0.7)(x1)
        x1 = Dense(5, activation='softmax')(x1)

        x = self.inception_block(x, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64)
        x = self.inception_block(x, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64)
        x = self.inception_block(x, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64)

        x2 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
        x2 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(x2)
        x2 = Flatten()(x2)
        x2 = Dense(1024, activation='relu')(x2)
        x2 = Dropout(0.7)(x2)
        x2 = Dense(1000, activation='softmax')(x2)

        x = self.inception_block(x, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = self.inception_block(x, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)
        x = self.inception_block(x, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4=128)

        x = GlobalAveragePooling2D(name='GAPL')(x)
        x = Dropout(0.4)(x)
        x = Dense(1000, activation='softmax')(x)

        model = Model(input_layer, [x, x1, x2], name='GoogLeNet')
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model


class ModelResNet34:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1000):
        self.model_name = "ResNet-34"
        self.input_shape = input_shape
        self.num_classes = num_classes
        print(f"Start generating model {self.model_name} generating!")
        self.model = self.construct_model("categorical_crossentropy", "adam")
        print(f"Model {self.model_name} generated!")

    @staticmethod
    def residual_block(input_tensor, filters, strides=(1, 1)):
        x = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        if strides != (1, 1) or input_tensor.shape[-1] != filters:
            input_tensor = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(input_tensor)
            input_tensor = BatchNormalization()(input_tensor)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def construct_model(self, loss, optimizer):
        input_tensor = Input(shape=self.input_shape)

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

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

        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=x)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model


def signal_to_spectrogram():
    # Using DFT to convert 1D signal array to spectrogram
    data = DataLoader(data_name="train",
                      batch_size=128,
                      extension="npy").get_data()
    dft = list()
    image = list()
    for i in range(len(data)):
        dft.append(np.abs(fft.fft(data[i])))
        image.append(np.reshape(dft[i], (1, len(data[i]))))
    for i in range(10):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].imshow(image[i], aspect=35, cmap="inferno")
        axs[1].plot(data[i])
        plt.show()


def generate_model_scheme(model, scale_xy: float = 1, scale_z: float = 1,
                          legend: bool = True, show: bool = False):
    scheme = layered_view(model.model,
                          to_file=f"model_figures/{model.model_name}.png",
                          legend=legend, scale_xy=scale_xy, scale_z=scale_z)
    if show:
        scheme.show()


def main():
    # train.main()
    # lenet5 = ModelLeNet5()
    # alexnet = ModelAlexNet()
    # vggnet = ModelVGGNetD()
    # googlenet = ModelGoogLeNet()
    # resnet34 = ModelResNet34()
    # generate_model_scheme(lenet5, 1, 0.1)
    # generate_model_scheme(alexnet, 1, 0.01)
    # generate_model_scheme(vggnet, 0.4, 0.001)
    # generate_model_scheme(googlenet, 1, 0.0001)
    # generate_model_scheme(resnet34, 1, 0.0001)
    pass


if __name__ == "__main__":
    main()
