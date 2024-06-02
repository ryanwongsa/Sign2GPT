# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of original dinov2 source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import torch

from torch import Tensor
from torch import nn
import math

logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        adapt: bool = False,
        adapt_params={},
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.adapt = adapt
        if self.adapt:
            if not adapt_params["w_lora"]:
                self.adapt = False
            else:
                lora_rank = adapt_params["lora_rank"]
                lora_a = adapt_params["lora_a"]
                lora_drop = adapt_params["lora_drop"]

                if "new" not in adapt_params:
                    self.lora_w1_l1 = nn.Linear(dim, lora_rank, bias=False)
                    self.lora_w1_l2 = nn.Linear(lora_rank, dim * 3, bias=False)
                    self.lora_w2_l1 = nn.Linear(dim, lora_rank, bias=False)
                    self.lora_w2_l2 = nn.Linear(lora_rank, dim, bias=False)
                    self.lora_scaling = lora_a / lora_rank
                    nn.init.normal_(self.lora_w1_l1.weight.data, 0, std=0.02)
                    nn.init.normal_(self.lora_w2_l1.weight.data, 0, std=0.02)
                    nn.init.constant_(self.lora_w1_l2.weight.data, 0)
                    nn.init.constant_(self.lora_w2_l2.weight.data, 0)
                    self.new_lora=False
                else:
                    self.lora_wq_l1 = nn.Linear(dim, lora_rank, bias=False)
                    self.lora_wq_l2 = nn.Linear(lora_rank, dim, bias=False)
                    self.lora_wk_l1 = nn.Linear(dim, lora_rank, bias=False)
                    self.lora_wk_l2 = nn.Linear(lora_rank, dim, bias=False)
                    self.lora_wv_l1 = nn.Linear(dim, lora_rank, bias=False)
                    self.lora_wv_l2 = nn.Linear(lora_rank, dim, bias=False)
                    self.lora_wo_l1 = nn.Linear(dim, lora_rank, bias=False)
                    self.lora_wo_l2 = nn.Linear(lora_rank, dim, bias=False)


                    self.lora_scaling = lora_a / lora_rank
                    nn.init.normal_(self.lora_wq_l1.weight.data, 0, std=0.02)
                    nn.init.normal_(self.lora_wk_l1.weight.data, 0, std=0.02)
                    nn.init.normal_(self.lora_wv_l1.weight.data, 0, std=0.02)
                    nn.init.normal_(self.lora_wo_l1.weight.data, 0, std=0.02)
                    nn.init.constant_(self.lora_wq_l2.weight.data, 0)
                    nn.init.constant_(self.lora_wk_l2.weight.data, 0)
                    nn.init.constant_(self.lora_wv_l2.weight.data, 0)
                    nn.init.constant_(self.lora_wo_l2.weight.data, 0)
                    self.new_lora=True
                self.lora_drop = nn.Dropout(lora_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        if self.adapt:
            if not self.new_lora:
                qkv = (
                    (
                        self.qkv(x)
                        + (
                            self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x)))
                            * self.lora_scaling
                        )
                    )
                    .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                qkv = (
                        self.qkv(x))
                qlora = (
                    self.lora_wq_l2(self.lora_wq_l1(self.lora_drop(x)))
                    * self.lora_scaling
                )
                klora = (
                    self.lora_wk_l2(self.lora_wk_l1(self.lora_drop(x)))
                    * self.lora_scaling
                )

                vlora = (
                    self.lora_wv_l2(self.lora_wv_l1(self.lora_drop(x)))
                    * self.lora_scaling
                )
                qkv = qkv+torch.cat([qlora, klora,vlora],dim=-1) 
                qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.adapt:
            if not self.new_lora:
                x = self.proj(x) + (
                    self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(x))) * self.lora_scaling
                )
            else:
                x = self.proj(x) + (
                    self.lora_wo_l2(self.lora_wo_l1(self.lora_drop(x))) * self.lora_scaling
                )
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        if self.adapt:
            if not self.new_lora:
                qkv = (
                    self.qkv(x)
                    + (
                        self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x)))
                        * self.lora_scaling
                    )
                ).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            else:
                qkv = (
                    self.qkv(x)
                )
                qlora = (
                    self.lora_wq_l2(self.lora_wq_l1(self.lora_drop(x)))
                    * self.lora_scaling
                )
                klora = (
                    self.lora_wk_l2(self.lora_wk_l1(self.lora_drop(x)))
                    * self.lora_scaling
                )

                vlora = (
                    self.lora_wv_l2(self.lora_wv_l1(self.lora_drop(x)))
                    * self.lora_scaling
                )
                qkv = qkv+torch.cat([qlora, klora,vlora],dim=-1)
                qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads) 

        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])
        if self.adapt:
            if not self.new_lora:
                x = self.proj(x) + (
                    self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(x))) * self.lora_scaling
                )
            else:
                x = self.proj(x) + (
                    self.lora_wo_l2(self.lora_wo_l1(self.lora_drop(x))) * self.lora_scaling
                )
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        return x
