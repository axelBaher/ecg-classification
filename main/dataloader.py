import json
import numpy as np
import os.path as osp
import wfdb
from scipy.signal import find_peaks
from sklearn.preprocessing import scale


ANNOTATION_PATH = {"train": "../data/train.json",
                   "valid": "../data/validation.json"}
MAPPING_PATH = "../data/class-mapper.json"  # C:/Bachelor-diploma/Code/data/class-mapper.json


class DataLoader:
    def __init__(self, data_name, batch_size):
        self.data_name = data_name
        self.data = json.load(open(ANNOTATION_PATH[data_name]))
        self.mapper = json.load(open(MAPPING_PATH))
        self.batch_size = batch_size
        self.samples_num = int(np.ceil(len(self.data) / batch_size))


def main():
    train = DataLoader("train", 128)
    valid = DataLoader("valid", 128)
    pass


if __name__ == "__main__":
    main()
