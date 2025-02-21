from argparse import ArgumentParser
import subprocess
import shutil
from pathlib import Path

import re
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from time import time
import io
import lmdb
import pickle

parser = ArgumentParser(parents=[])

parser.add_argument(
    "--id",
    type=str,
    default="S001843_P0005_T00",
)

parser.add_argument(
    "--data_dir",
    type=str,
    default="dataset_creation/csl-daily",
)

parser.add_argument(
    "--lmdb_dir",
    type=str,
    default="csl-daily/lmdb_videos",
)

params, unknown = parser.parse_known_args()

id = params.id
data_dir = Path(params.data_dir)
lmdb_dir = Path(params.lmdb_dir) / id

video_path = str(data_dir / f"{id}.mp4")

n_bytes = 2**40

tmp_dir = Path("/tmp") / f"TEMP_{time()}"
env = lmdb.open(path=str(tmp_dir), map_size=n_bytes)
txn = env.begin(write=True)

cap = cv2.VideoCapture(video_path)

if lmdb_dir.exists() and lmdb_dir.is_dir():
    exit()

lmdb_dir.mkdir(parents=True, exist_ok=True)

ind = 0
counter = 0
while True:
    ret = cap.grab()
    if not ret:
        break
    ret, frame = cap.retrieve()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame).resize((256, 256))
    temp = io.BytesIO()
    img.save(temp, format="jpeg")
    temp.seek(0)
    txn.put(
        key=f"{ind}".encode("ascii"),
        value=temp.read(),
        dupdata=False,
    )
    ind += 1
    counter += 1

    if counter % 123 == 0 and counter != 0:
        txn.commit()
        txn = env.begin(write=True)

txn.put(
    key=f"details".encode("ascii"),
    value=pickle.dumps({"num_frames": ind, "id": id}, protocol=4),
    dupdata=False,
)
txn.commit()

env.close()

if lmdb_dir.exists():
    shutil.rmtree(lmdb_dir)
shutil.move(f"{tmp_dir}", f"{lmdb_dir}")
