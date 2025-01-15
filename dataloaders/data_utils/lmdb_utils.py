import lmdb
from PIL import Image
import numpy as np
import io
import pickle
from collections import defaultdict
import torch

class LMDBUtility(object):
    def __init__(self, lmdb_dir, landmark_keys={}):
        self.lmdb_dir = lmdb_dir

        with lmdb.open(
            path=f"{lmdb_dir}",
            readonly=True,
            readahead=False,
            lock=False,
            meminit=False,
        ).begin(write=False) as txn:
            self.details = pickle.loads(txn.get(key=f"details".encode("ascii")))
        self.landmark_keys = landmark_keys

    def get_frames(self, selection):
        with lmdb.open(
            path=f"{self.lmdb_dir}",
            readonly=True,
            readahead=False,
            lock=False,
            meminit=False,
        ).begin(write=False) as txn:
            frames = [
                np.array(Image.open(io.BytesIO(txn.get(key=f"{idx}".encode("ascii")))))
                for idx in selection
            ]
        return frames