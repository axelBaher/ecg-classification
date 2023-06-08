import json
import numpy as np
import cv2
from tqdm import tqdm

ANNOTATION_PATH = {"train": "../data/train.json",
                   "test": "../data/test.json"}
MAPPING_PATH = "../data/class-mapper.json"


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
                for elem, i in tqdm(zip(self.data, range(len(self.data))), total=len(self.data), desc="Files read"):
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
