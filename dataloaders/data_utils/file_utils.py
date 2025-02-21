import pickle
import gzip


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

import json
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data