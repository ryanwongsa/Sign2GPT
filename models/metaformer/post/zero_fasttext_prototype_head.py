import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import importlib

import fasttext.util
from dataloaders.data_utils.file_utils import read_pickle
import ignite.distributed as idist

class HeadModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        emb_lang,
        emb_pkl_dir,
        trainable_emb,
        dropout=0.5,
        class_temperature=0.1,
        time_temperature=0.1,
        dynamic_time_temperatures=False,
        dynamic_class_temperatures=False,
    ):
        super().__init__()
        if not dynamic_time_temperatures:
            self.register_buffer("time_temperature", torch.tensor(time_temperature))
        else:
            self.time_temperature = torch.nn.Parameter(torch.ones(1) * time_temperature)
        if not dynamic_class_temperatures:
            self.register_buffer("class_temperature", torch.tensor(class_temperature))
        else:
            self.class_temperature = torch.nn.Parameter(
                torch.ones(1) * class_temperature
            )

        if hidden_dim is not None:
            self.fc_hidden = nn.Linear(in_dim, hidden_dim)
        else:
            self.fc_hidden = nn.Identity()
            hidden_dim = in_dim

        if idist.get_local_rank() == 0 or idist.get_world_size() == 0:
            fasttext.util.download_model(emb_lang, if_exists="ignore")
        if idist.get_world_size() > 0:
            idist.barrier()
        ft = fasttext.load_model(f"cc.{emb_lang}.300.bin")

        dict_processed_words = read_pickle(emb_pkl_dir) # DICTIONARY CONTAINING THE {<PSUEDOGLOSS>:<ID>}
        dict_lem_to_id = dict_processed_words["dict_lem_to_id"]
        vector = torch.zeros((len(dict_lem_to_id), 300))
        for key, value in dict_lem_to_id.items():
            vector[value] = torch.tensor(ft.get_word_vector(key))

        self.vocab_embedding = torch.nn.Parameter(vector.permute(1, 0).unsqueeze(-1))
        self.vocab_embedding.requires_grad = trainable_emb
        zero_embedding = torch.zeros(hidden_dim, 1, 1)
        self.register_buffer("zero_embedding", zero_embedding)

        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

    def logit_compare_embed(self, out, vocab, mask):
        N, T, C = out.shape

        vocab = torch.cat([vocab, self.zero_embedding], dim=1)
        _, V, M = vocab.shape

        out = F.normalize(out, dim=-1)

        vocab = F.normalize(vocab, dim=0)

        fc_out = torch.bmm(
            out.reshape(N, T, C),
            vocab.reshape(C, V * M).unsqueeze(0).repeat(N, 1, 1),
        ).reshape(N, T, V, M)

        fc_out = F.adaptive_avg_pool3d(fc_out, (T, V, 1)).squeeze(-1)

        logits = fc_out.reshape(N, T, V)

        return logits

    def forward(self, x, mask):
        b, t, c = x.shape

        y = self.fc_hidden(self.dropout(x))

        time_res = self.logit_compare_embed(y, self.vocab_embedding, mask)
        cls_temp = torch.clamp(self.class_temperature, 0.01, 1.0)
        cls_softmax = (time_res / cls_temp).softmax(axis=-1)
        time_mask = (
            (~mask)
            .type(time_res.dtype)
            .masked_fill(~mask, torch.finfo(time_res.dtype).min)
            .unsqueeze(-1)
        )
        time_temp = torch.clamp(self.time_temperature, 0.01, 1.0)
        time_softmax = (time_res / time_temp + time_mask).softmax(axis=-2)

        softmax_scores = cls_softmax * time_softmax
        class_scores = softmax_scores.sum(axis=-2)
        logits = class_scores[:, : self.num_classes]

        return {
            "time_res": time_res,
            "softmax_scores": softmax_scores,
            "logits": logits,
            "mask": mask,
        }
