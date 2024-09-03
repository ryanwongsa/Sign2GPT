from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np


class F1Score(Metric):
    def __init__(self, thresholds, output_transform=lambda x: x):
        self.tp = None
        self.fp = None
        self.fn = None
        self.thresholds = thresholds
        super(F1Score, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        super(F1Score, self).reset()

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output
        preds = preds.float().detach()

        gloss_targets = torch.zeros(
            preds.shape[0], preds.shape[-1], dtype=preds.dtype, device=preds.device
        )
        for i, t in enumerate(targets):
            for t_i in t:
                gloss_targets[i, t_i] = 1.0
        gloss_targets = gloss_targets.bool()
        pos = preds > self.thresholds
        match = gloss_targets == pos
        not_match = gloss_targets != pos

        self.tp += match[gloss_targets].sum()
        self.fp += not_match[~gloss_targets].sum()
        self.fn += not_match[gloss_targets].sum()

    @sync_all_reduce("tp", "fp", "fn")
    def compute(self):
        return self.tp / (self.tp + 0.5 * (self.fp + self.fn))
