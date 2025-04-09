from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np


class AccuracyScore(Metric):
    def __init__(self, thresholds, output_transform=lambda x: x):
        self.correct = None
        self.total = None
        self.thresholds = thresholds
        super(AccuracyScore, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.correct = torch.zeros(1)
        self.total = 0
        super(AccuracyScore, self).reset()

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
        correct = ((pos == gloss_targets) & (gloss_targets)).sum()
        self.correct += correct.cpu()
        self.total += gloss_targets.sum().item()

    @sync_all_reduce(
        "correct",
        "total",
    )
    def compute(self):
        accuracy = self.correct / self.total
        return accuracy.item()
