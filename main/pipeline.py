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
        "VGGNetD": m.ModelVGGNetD(),
        "GoogLeNet": m.ModelGoogLeNet(),
        "ResNet34": m.ModelResNet34()
    })
    return models


def main():

    prep.main()


if __name__ == "__main__":
    main()
