import os
# from visualkeras import layered_view
# from tensorflow.python.client import device_lib
import itertools


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# def signal_to_spectrogram():
#     # Using DFT to convert 1D signal array to spectrogram
#     data = DataLoader(data_name="train",
#                       batch_size=128,
#                       extension="npy").get_data()
#     dft = list()
#     image = list()
#     for i in range(len(data)):
#         dft.append(np.abs(fft.fft(data[i])))
#         image.append(np.reshape(dft[i], (1, len(data[i]))))
#     for i in range(10):
#         fig, axs = plt.subplots(2, 1, figsize=(8, 6))
#         axs[0].imshow(image[i], aspect=35, cmap="inferno")
#         axs[1].plot(data[i])
#         plt.show()
#
#
# def generate_model_scheme(model, scale_xy: float = 1, scale_z: float = 1,
#                           legend: bool = True, show: bool = False):
#     scheme = layered_view(model.model,
#                           to_file=f"model_figures/{model.model_name}.png",
#                           legend=legend, scale_xy=scale_xy, scale_z=scale_z)
#     if show:
#         scheme.show()
#
#
# def check_cuda_availability():
#     local_device_protos = device_lib.list_local_devices()
#     gpu_available = any(device.device_type == 'GPU' for device in local_device_protos)
#     cuda_available = any('GPU' in device.physical_device_desc for device in local_device_protos)
#
#     if gpu_available:
#         print("GPU is available in system.")
#     else:
#         print("GPU is not available in system.")
#
#     if cuda_available:
#         print("CUDA is available in system.")
#     else:
#         print("CUDA is not available in system.")


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
        # "GoogLeNet",
        # "ResNet34"
        # "VGGNetD"
    ])
    for i, model in enumerate(models):
        for j, param_config in enumerate(param_combs):
            index = (j + (len(param_combs) * i)) + 1
            print(f"{index}/{len(param_combs) * len(models)}")
#     check_cuda_availability()
#     # train.main()
#     # lenet5 = ModelLeNet5()
#     # alexnet = ModelAlexNet()
#     # vggnet = ModelVGGNetD()
#     # googlenet = ModelGoogLeNet()
#     # resnet34 = ModelResNet34()
#     # generate_model_scheme(lenet5, 1, 0.1)
#     # generate_model_scheme(alexnet, 1, 0.01)
#     # generate_model_scheme(vggnet, 0.4, 0.001)
#     # generate_model_scheme(googlenet, 1, 0.0001)
#     # generate_model_scheme(resnet34, 1, 0.0001)
#     pass
#

if __name__ == "__main__":
    main()
