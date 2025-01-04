from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np


class ClassF1Score(Metric):
    def __init__(self, thresholds, num_classes, output_transform=lambda x: x):
        self.tp = None
        self.fp = None
        self.fn = None
        self.thresholds = thresholds
        self.num_classes = num_classes
        super(ClassF1Score, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        super(ClassF1Score, self).reset()

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

        self.tp += (match & gloss_targets).sum(axis=0).detach().cpu()
        self.fp += (not_match & ~gloss_targets).sum(axis=0).detach().cpu()
        self.fn += (not_match & gloss_targets).sum(axis=0).detach().cpu()

    @sync_all_reduce("tp", "fp", "fn")
    def compute(self):
        total = self.tp + self.fn
        return (
            self.tp[total > 0]
            / (self.tp[total > 0] + 0.5 * (self.fp[total > 0] + self.fn[total > 0]))
        ).mean()
