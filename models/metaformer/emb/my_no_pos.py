# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# Silence Mypy errors in this file.
# type: ignore

import math

import torch

from xformers.components.positional_embedding import (
    PositionEmbedding,
    PositionEmbeddingConfig,
    register_positional_embedding,
)


@register_positional_embedding("my_no_pos", PositionEmbeddingConfig)
class MyNoPositionalEmbedding(PositionEmbedding):
    def __init__(self, dim_model: int, max_len=2048, *args, **kwargs):
        super().__init__()
        self.dim_model = dim_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        return output
