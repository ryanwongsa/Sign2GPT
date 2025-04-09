from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np


class ClassAccuracyScore(Metric):
    def __init__(
        self, thresholds, num_classes, output_transform=lambda x: x
    ):
        self.correct = None
        self.total = None
        self.num_classes = num_classes
        self.thresholds = thresholds

        super(ClassAccuracyScore, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.correct = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)
        super(ClassAccuracyScore, self).reset()

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
        correct = (pos == gloss_targets) & (gloss_targets)
        self.correct += correct.sum(axis=0).cpu()
        self.total += gloss_targets.sum(axis=0).cpu()

    @sync_all_reduce(
        "correct",
        "total",
    )
    def compute(self):
        accuracy = self.correct / (self.total + 1e-8)
        return (accuracy[self.total > 0]).mean().item()
