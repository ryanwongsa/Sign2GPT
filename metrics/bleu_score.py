from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np
from metrics.utils.sacrebleu import (
    compute_bleu,
    SMOOTH_VALUE_DEFAULT,
    corpus_bleu_attributes,
)


class BLEUScore(Metric):
    def __init__(self, pre_name="", output_transform=lambda x: x):
        self.correct1 = None
        self.total1 = None

        self.correct2 = None
        self.total2 = None

        self.correct3 = None
        self.total3 = None

        self.correct4 = None
        self.total4 = None

        self.sys_len = None
        self.ref_len = None
        self.pre_name = pre_name

        super(BLEUScore, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.correct1 = 0
        self.total1 = 0

        self.correct2 = 0
        self.total2 = 0

        self.correct3 = 0
        self.total3 = 0

        self.correct4 = 0
        self.total4 = 0

        self.sys_len = 0
        self.ref_len = 0
        super(BLEUScore, self).reset()

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output
        # targets is a list ["a is an apple","b is a banana"]
        # pred is a list ["what is an apple", "b is a banana"]
        # beginning of sacreableu: corpus_bleu method
        dict_bleu_attributes = corpus_bleu_attributes(
            sys_stream=preds,
            ref_streams=[targets],
            smooth_method="floor",
            smooth_value=SMOOTH_VALUE_DEFAULT,
            force=True,
            tokenize="none",
            use_effective_order=True,
        )

        self.correct1 += dict_bleu_attributes["correct"][0]
        self.total1 += dict_bleu_attributes["total"][0]

        self.correct2 += dict_bleu_attributes["correct"][1]
        self.total2 += dict_bleu_attributes["total"][1]

        self.correct3 += dict_bleu_attributes["correct"][2]
        self.total3 += dict_bleu_attributes["total"][2]

        self.correct4 += dict_bleu_attributes["correct"][3]
        self.total4 += dict_bleu_attributes["total"][3]

        self.sys_len += dict_bleu_attributes["sys_len"]
        self.ref_len += dict_bleu_attributes["ref_len"]

    @sync_all_reduce(
        "correct1",
        "correct2",
        "correct3",
        "correct4",
        "total1",
        "total2",
        "total3",
        "total4",
        "sys_len",
        "ref_len",
    )
    def compute(self):
        # end of sacreableu: corpus_bleu method
        # print([self.correct1, self.correct2, self.correct3, self.correct4],[self.total1, self.total2, self.total3, self.total4], self.sys_len, self.ref_len)
        bleu_scores = compute_bleu(
            [self.correct1, self.correct2, self.correct3, self.correct4],
            [self.total1, self.total2, self.total3, self.total4],
            self.sys_len,
            self.ref_len,
            smooth_method="floor",
            smooth_value=SMOOTH_VALUE_DEFAULT,
            use_effective_order=True,
        ).scores
        scores = {}
        for n in range(len(bleu_scores)):
            if self.pre_name != "":
                scores[self.pre_name + "_bleu" + str(n + 1)] = bleu_scores[n]
            else:
                scores["bleu" + str(n + 1)] = bleu_scores[n]
        return scores