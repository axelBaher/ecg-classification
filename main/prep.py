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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


INSTALL_DB = 0
ARCHIVE_PATH = "../mit-bih.zip"
NUMBER_OF_RECORDS = 5


def download_db():
    # Downloading database
    db_exists = os.path.exists("../mit-bih")
    if INSTALL_DB and (not db_exists):
        url = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading mit-bih database") as t:
            urllib.request.urlretrieve(url, filename=ARCHIVE_PATH, reporthook=lambda b, bsize, _: t.update(b))
        with zipfile.ZipFile(ARCHIVE_PATH, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            extracted_files = 0
            with tqdm(total=total_files, desc="Archive unzipping") as pbar:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, "../")
                    extracted_files += 1
                    pbar.update(1)
        os.rename("../mit-bih-arrhythmia-database-1.0.0", "../mit-bih")
        os.remove(ARCHIVE_PATH)


def get_records_name(db_path):
    # Getting names of the records
    pattern = re.compile(r"\d+")
    names = glob(f"{db_path}\\*.dat")
    records = list()
    for name in names:
        match = pattern.search(name)
        if match:
            record = str(match.group())
            records.append(record)
    return records


def generate_data():
    db_path = "../mit-bih"
    records = get_records_name(db_path)
    slash = jg.get_slash()
    for record in records[:NUMBER_OF_RECORDS - 1]:
        path_1D = slash.join(["..", "data", "1D", record])
        path_2D = slash.join(["..", "data", "2D", record])
        if not (os.path.exists(path_1D) and os.path.exists(path_2D)):
            dg.data_gen(db_path + slash + record)
        else:
            print(f"Data generation of {record} record already done!")
    jg.json_gen()


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
    download_db()
    generate_data()
