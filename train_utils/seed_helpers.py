import torch
import random
import numpy as np
import os


def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False # SINCE VARYING INPUT
