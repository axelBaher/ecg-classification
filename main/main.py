# import train
from dataloader import DataLoader
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, AveragePooling2D, MaxPooling2D, Flatten, Dropout
from visualkeras import layered_view
import pydot


class ModelLeNet5:
    def __init__(self):
        self.model_name = "LeNet-5"
        self.model = self.construct_model("categorical_crossentropy", "adam")

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
        self.model = self.construct_model("categorical_crossentropy", "adam")

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


"""
class Model:
    def __init__(self):
        self.model_name = ""
        self.model = self.construct_model("categorical_crossentropy", "adam")

    @staticmethod
    def construct_model(loss, optimizer):
        model = Sequential()
        return model
"""


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


def generate_model_figure(model):
    layered_view(model.model, to_file=f"model_figures/{model.model_name}.png",
                 legend=True)


def main():
    # train.main()
    # signal_to_spectrogram()
    # LeNet5 = ModelLeNet5()
    # AlexNet = ModelAlexNet()
    # generate_model_figure(ModelLeNet5())

    # model = ModelLeNet5()
    # model.save_model_figure()
    layered_view(ModelLeNet5().model, to_file=f"model_figures/{ModelLeNet5().model_name}.png",
                 legend=True)
    layered_view(ModelAlexNet().model, to_file=f"model_figures/{ModelAlexNet().model_name}.png",
                 legend=True)

    # file = Image.open(f"{model.model_name}.png")
    # model_image.show()
    # file.save(f"model_figures/{model.model_name}.png")
    pass


if __name__ == "__main__":
    main()
