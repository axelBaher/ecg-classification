import json
import numpy as np
import os.path as osp
import wfdb
from scipy.signal import find_peaks
from sklearn.preprocessing import scale
import cv2
from tensorflow import float16

ANNOTATION_PATH = {"train": "../data/train.json",
                   "valid": "../data/validation.json"}
MAPPING_PATH = "../data/class-mapper.json"  # C:/Bachelor-diploma/Code/data/class-mapper.json


class DataLoader:
    def __init__(self, data_name, batch_size, extension):
        self.extension = extension
        self.data_name = data_name
        self.data = json.load(open(ANNOTATION_PATH[data_name]))
        self.mapper = json.load(open(MAPPING_PATH))
        self.batch_size = batch_size
        self.samples_num = int(np.ceil(len(self.data) / batch_size))

    def get_data(self):
        data = list()
        match self.extension:
            case "npy":
                for elem, i in zip(self.data, range(len(self.data))):
                    print(f"{i}/{len(self.data)} files read", end='\r')
                    data.append(np.load(elem["path"]))
            case "png":
                for elem, i in zip(self.data, range(len(self.data))):
                    print(f"{i}/{len(self.data)} files read", end='\r')
                    img = np.float16(cv2.imread(elem["path"], cv2.IMREAD_GRAYSCALE))
                    data.append(img)
        # data = np.array_split(np.array(data), self.samples_num)
        return data

    def data_split(self):
        def filter_func(a, b): return a["label"] == b

        data_splitted = list()
        for label in self.mapper.keys():
            data_splitted.append(list(
                filter(
                    lambda x: filter_func(x, label),
                    self.data)))
        return data_splitted

# def main():
#     # train = DataLoader(128)
#     # valid = DataLoader(128)
#     pass
#
#
# if __name__ == "__main__":
#     main()
