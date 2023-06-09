from glob import glob
import re
import os
import data_generation as dg
import json_generation as jg
import models as m
import urllib.request
import zipfile
from tqdm import tqdm
# import train
# import dataloader
import prep


def generate_models():
    models = dict({
        "LeNet5": m.ModelLeNet5(),
        "AlexNet": m.ModelAlexNet(),
        "GoogLeNet": m.ModelGoogLeNet(),
        "ResNet34": m.ModelResNet34(),
        "VGGNetD": m.ModelVGGNetD()
    })
    return models


def main():
    epoch = list([1, 2, 3])
    batch_size = list([16, 32, 64, 128, 256, 512])
    models = generate_models()
    for model in models:
        # call train with epoch and batch_size parameters
        pass


if __name__ == "__main__":
    main()
