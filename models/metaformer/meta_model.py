import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import importlib
import math
import numpy as np


class MetaFormer(nn.Module):
    def __init__(
        self, emb_name, emb_params, net_name, net_params, post_name, post_params, inits='standard'
    ):
        super().__init__()
        emb_mod = importlib.import_module(emb_name, package=None)
        net_mod = importlib.import_module(net_name, package=None)
        post_mod = importlib.import_module(post_name, package=None)

        self.embedding_block = emb_mod.Model(**emb_params)
        self.network_block = net_mod.Model(**net_params)
        self.post_block = post_mod.Model(**post_params)

        self.num_layers = np.array(net_params["layers"]).sum()
        self.inits = inits
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear)):
            if self.inits == 'standard':
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif self.inits =='xavier':
                torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if "token_mixer.proj" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if self.inits == 'standard':
                    p.data.normal_(
                        mean=0.0,
                        std=(0.02 / math.sqrt(2 * self.num_layers)),
                    )
                # elif self.inits == 'xavier':
                #     torch.nn.init.xavier_uniform_(p.data)

    def forward_embeddings(self, x, mask):
        x, mask = self.embedding_block(x, mask)
        return x, mask

    def forward_tokens(self, x, mask, return_features=False):
        if return_features:
            x, mask, features = self.network_block(
                x, mask, return_features=return_features
            )
            return x, mask, features
        else:
            x, mask = self.network_block(x, mask, return_features=return_features)
            return x, mask

    def forward_post(self, x, mask):
        dict_post = self.post_block(x, mask)
        return dict_post #x, mask

    def forward(self, x, mask):
        list_of_features = []
        x, mask = self.forward_embeddings(x, mask)

        list_of_features.append((x, mask))
        x, mask, lf = self.forward_tokens(x, mask, return_features=True)
        list_of_features.extend(lf)
        hidden_state = x
        hidden_mask = mask
        post_output = self.forward_post(x, mask)

        return {
            "post_output": post_output,
            "hidden_state": hidden_state,
            "hidden_mask": hidden_mask,
            "list_of_features": list_of_features
        }

