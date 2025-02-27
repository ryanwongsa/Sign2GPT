from absl import app
import importlib
from absl import flags
import torch
import ignite.distributed as idist
from pathlib import Path
import os
from argparse import ArgumentParser

from ml_collections import config_flags
import random

import os

os.environ["TRITON_CACHE_DIR"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
torch.hub.set_dir("/tmp")


def main(_):
    mod = importlib.import_module(CONFIG.value.main_runner, package=None)
    backend = "nccl"
    nproc_per_node = (
        torch.cuda.device_count() if torch.cuda.device_count() > 1 else None
    )

    with idist.Parallel(
        backend, nproc_per_node=nproc_per_node, # master_port=random.randint(49152, 65535)
    ) as parallel:
        parallel.run(mod.Trainer, CONFIG.value)


if __name__ == "__main__":
    CONFIG = config_flags.DEFINE_config_file(
        "config",
        default="",
    )
    app.run(main)
