from dataloaders.data_utils.my_concat_dataset import MyConcatDataset
from tqdm import tqdm
import ignite.distributed as idist
import pandas as pd
import json
import pickle
import importlib


def get_dataset(ds_name, ds_params={}, transform=None):
    mod = importlib.import_module(ds_name, package=None)
    return mod.get_ds(ds_params, transform=transform)
