# import train
import os
from dataloader import DataLoader
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
from visualkeras import layered_view
from tensorflow.python.client import device_lib
from keras import Model
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, Dense, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Input, concatenate, \
    GlobalAveragePooling2D, BatchNormalization, Activation, Add
import pydot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def check_cuda_availability():
    local_device_protos = device_lib.list_local_devices()
    gpu_available = any(device.device_type == 'GPU' for device in local_device_protos)
    cuda_available = any('GPU' in device.physical_device_desc for device in local_device_protos)

    if gpu_available:
        print("GPU is available in system.")
    else:
        print("GPU is not available in system.")

    if cuda_available:
        print("CUDA is available in system.")
    else:
        print("CUDA is not available in system.")


def main():
    # check_cuda_availability()
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
