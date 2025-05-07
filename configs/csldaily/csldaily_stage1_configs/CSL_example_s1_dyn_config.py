from ml_collections import config_dict
from pathlib import Path
import os
import json
import numpy as np
import torch
from configs.base.base_utils import *


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.name = Path(os.path.realpath(__file__)).stem
    base_name = Path(os.path.realpath(__file__)).parent.name
    code_path = str(Path(os.path.realpath(__file__)).resolve().parents[3])

    ckpt_path = get_checkpoint_path(base_name, cfg.name)
    lmdb_path = get_lmdb_path()
    cfg.save_dir = f"{ckpt_path}/{base_name}/{cfg.name}"

    cfg.main_runner = "trainer.psuedo_gloss_trainer"
    cfg.project_name = "final_csl_pretrain"
    cfg.aug_name = "augmentation.video.base_video_aug"

    cfg.aug_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,
        "max_seq_len": 256,
    }

    base_bs = 16
    cfg.bs = int(
        base_bs
        * torch.cuda.device_count()
        # * ((torch.cuda.mem_get_info()[1] / 10**6) / 24000)
    )
    cfg.accum = 1
    cfg.num_workers = min(min(cfg.bs, int(10 * torch.cuda.device_count())), 10)

    cfg.gate_grad_multiplier = 1.0

    cfg.lr = 3e-4 # * (cfg.bs**0.5) / (base_bs**0.5)
    cfg.lr_scheduler = "warmupwithcosine"
    cfg.lr_scheduler_params = config_dict.ConfigDict(
        {
            "lr_scale_factor": 0.01,
            "num_cycles": 1,
            "start_value_mult": 0.7,
            "end_value_mult": 0.7,
            "warmup_epochs": 5,
        }
    )

    cfg.optimizer_name = "adamw"
    cfg.optimizer_params = config_dict.ConfigDict(
        {
            "lr": cfg.lr,
            "weight_decay": 0.001,
        } 
    )

    cfg.criterion_name = "losses.base_loss"

    cfg.criterion_params = config_dict.ConfigDict(
        {
            "dict_of_loss_params": {
                "bce": {
                    "cls_name": "losses.loss_functions.bce_loss",
                    "loss_params": {
                        "src_name": (
                            "dict_post_output",
                            "logits",
                        ),
                        "tgt_name": "pseudo_gloss_ids",
                    },
                    "weight": 10.0,
                },
            }
        }
    )

    cfg.max_epochs = 100
    cfg.model_checkpoint_dir = ""

    cfg.train_ds_name = "dataloaders.csldaily_video_dataset"
    cfg.valid_ds_name = "dataloaders.csldaily_video_dataset"
    cfg.test_ds_name = "dataloaders.csldaily_video_dataset"
    train_ds_params = {
        "split": "train",
        "pseudo_gloss_dir": f"{code_path}/data/csldaily/processed_words.csl_pkl",
        "tsv_dir": f"{code_path}/data/csldaily/data.tsv",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/combined_dataset/csl-daily/lmdb_videos",
            "isValid": False,
        },
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": True,
    }
    cfg.train_ds_params = config_dict.ConfigDict(train_ds_params)

    valid_ds_params = {
        "split": "dev",
        "pseudo_gloss_dir": f"{code_path}/data/csldaily/processed_words.csl_pkl",
        "tsv_dir": f"{code_path}/data/csldaily/data.tsv",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/combined_dataset/csl-daily/lmdb_videos",
            "isValid": True,
        },
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": False,
    }
    cfg.valid_ds_params = config_dict.ConfigDict(valid_ds_params)

    test_ds_params = {
        "split": "test",
        "pseudo_gloss_dir": f"{code_path}/data/csldaily/processed_words.csl_pkl",
        "tsv_dir": f"{code_path}/data/csldaily/data.tsv",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/combined_dataset/csl-daily/lmdb_videos",
            "isValid": True,
        },
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": False,
    }
    cfg.test_ds_params = config_dict.ConfigDict(test_ds_params)


    from configs.standards.standard_meta_model_zero_config import get_sign_encoder
    model_name, sign_model_params, dim_model = get_sign_encoder()
    cfg.model_name = model_name
    post_params = {
        "in_dim": dim_model,
        "hidden_dim": 300,
        "num_classes": 7918,
        "dropout": 0.2,
        "class_temperature": 0.1,
        "time_temperature": 0.1,
        "dynamic_time_temperatures": True,
        "dynamic_class_temperatures": True,
        "emb_lang": "zh",
        "emb_pkl_dir": f"{code_path}/data/csldaily/processed_words.csl_pkl",
        "trainable_emb": False,
    }

    cfg.model_params = {
        "sign_model_name": "models.model_sign_encoder.basic_sign_encoder",
        "sign_model_params": sign_model_params,
        "post_name": "models.metaformer.post.zero_fasttext_prototype_head",
        "post_params": post_params,
    }

    cfg.seed = 1
    cfg.grad_clip_norm = 1.0
    cfg.grad_clip_value = 1.0
    cfg.logger_name = ["text"] #["wandb"]
    cfg.resume = True
    cfg.train_length = None
    cfg.val_length = None
    cfg.log_every = 100
    cfg.save_ckpt = True
    cfg.score_factor = 1
    cfg.score_name = "valid/class_f1_score"
    cfg.bfloat16_only = False

    return cfg
