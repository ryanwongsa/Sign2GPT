import os
import socket


CONFIG = {
    "WANDB_API_KEY": "<YOUR_WANDB_API_KEY>",
    "WANDB_IGNORE_GLOBS": "*.patch",
    "WANDB_DISABLE_CODE": "true",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_DIR": "/tmp",
    "WANDB_CACHE_DIR": "/tmp",
    "WANDB_CONFIG_DIR": "/tmp",
    # "CUDA_LAUNCH_BLOCKING": "1", # ADD IF ISSUES WITH CUDA
    "WANDB_HOST": (
        f"{socket.gethostname()}"
        if os.getenv("SLURM_JOB_ID") is None
        else f"{os.getenv('SLURM_ARRAY_JOB_ID')}_{os.getenv('SLURM_ARRAY_TASK_ID')}-{socket.gethostname()}"
    ),
}
