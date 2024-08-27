import torch
import torch.nn as nn
import xformers.components.positional_embedding as PE
from xformers.components.positional_embedding import build_positional_embedding
import models.metaformer.emb.my_no_pos
import models.metaformer.emb.my_sine


class Model(nn.Module):
    def __init__(self, in_dim, d_model, pos_config, num_embed_tokens = None):
        super().__init__()
        if in_dim != d_model:
            self.linear = nn.Linear(in_dim, d_model)
        else:
            self.linear = nn.Identity()
        self.pos_emb = build_positional_embedding(pos_config)
        self.num_embed_tokens = num_embed_tokens
        if num_embed_tokens:
            self.embed_tokens = torch.nn.Embedding(num_embed_tokens, d_model)
            self.embed_tokens.weight.data.normal_(mean=0,std=0.7)
        else:
            self.embed_tokens = None


    def forward(self, x, mask):
        if self.embed_tokens is not None:
            embeds = self.embed_tokens.weight
            # ADDED [:,:x.shape[1]] to keep it at 2** shape, shouldn't impact results as padding is large
            mask = torch.cat([torch.ones(mask.shape[0],self.num_embed_tokens).bool().to(mask.device), mask],dim=1)[:,:x.shape[1]]
            x = torch.cat([embeds.unsqueeze(0).expand(mask.shape[0], *embeds.shape), x], dim=1)[:,:x.shape[1],:]
        
        x = self.linear(x)
        x = self.pos_emb(x)
        return x, mask
