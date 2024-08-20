import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, src_name, tgt_name):
        super().__init__()

        self.src_name = src_name
        self.tgt_name = tgt_name
        self.bce_loss_fn = torch.nn.BCELoss(reduction="none")

    def forward(self, y_pred, target):
        if isinstance(self.src_name, tuple):
            src = y_pred
            for mn in self.src_name:
                src = src[mn]
        else:
            src = y_pred[self.src_name]
        tgt = target[self.tgt_name]

        gloss_targets = torch.zeros(
            src.shape[0], src.shape[-1], dtype=src.dtype, device=src.device
        )
        for i, t in enumerate(tgt):
            for t_i in t:
                gloss_targets[i, t_i] = 1.0

        loss = self.bce_loss_fn(torch.clamp(src, 1e-8, 1 - 1e-8), gloss_targets)

        return loss.mean()


if __name__ == "__main__":
    loss_params = {}
    loss_fn = Loss(**loss_params)

    dict_preds = {
        "target": {},
        "y_pred": {},
    }

    loss = loss_fn(**dict_preds)
