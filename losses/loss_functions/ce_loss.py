import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, src_name, tgt_name, mask_name, label_smoothing=0.0):
        super().__init__()

        self.src_name = src_name
        self.tgt_name = tgt_name
        self.mask_name = mask_name
        self.ce_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction="none"
        )

    def forward(self, y_pred, target):
        mask = target[self.mask_name]
        src = y_pred[self.src_name][mask]

        tgt = target[self.tgt_name][mask]

        return self.ce_loss_fn(src, tgt).mean()


if __name__ == "__main__":
    loss_params = {}
    loss_fn = Loss(**loss_params)

    dict_preds = {
        "target": {},
        "y_pred": {},
    }

    loss = loss_fn(**dict_preds)
