import torch
from torch import nn
import ignite.distributed as idist
import numpy as np
import torch.nn.functional as F
import importlib


class Model(nn.Module):
    def __init__(
        self,
        spatial_name,
        spatial_params,
        encoder_name,
        encoder_params,
    ):
        super().__init__()

        spatial_mod = importlib.import_module(spatial_name, package=None)
        spatial_params["out_dim"] = encoder_params["emb_params"]["in_dim"]
        self.spatial_model = spatial_mod.Model(**spatial_params)

        encoder_mod = importlib.import_module(encoder_name, package=None)
        self.encoder = encoder_mod.MetaFormer(**encoder_params)

    def forward(
        self,
        frame_features,
        max_len,
    ):
        x, mask, dict_feat = self.spatial_model(frame_features, max_len=max_len)
        enc_output = self.encoder(x, mask)
        return {
            "enc_output": enc_output,
            "dict_feat": dict_feat,
        }
