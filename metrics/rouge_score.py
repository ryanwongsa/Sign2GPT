from metrics.utils.mscoco_rouge import calc_score

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np


class RougeMetric(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(RougeMetric, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.n_seq = 0
        self.rouge_score = 0
        super(RougeMetric, self).reset()

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output
        # targets is a list ["a is an apple","b is a banana"]
        # pred is a list ["what is an apple", "b is a banana"]
        hypotheses = preds
        references = targets
        for h, r in zip(hypotheses, references):
            self.rouge_score += calc_score(hypotheses=[h], references=[r])
            self.n_seq += 1

    @sync_all_reduce(
        "n_seq",
        "rouge_score",
    )
    def compute(self):
        score = self.rouge_score / self.n_seq
        return score * 100