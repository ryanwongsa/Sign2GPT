# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional
import math
from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        adapt: bool = False,
        adapt_params={},
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

        self.adapt = adapt
        if self.adapt:
            if not adapt_params["w_lora_ff"]:
                self.adapt = False
            else:
                lora_rank = adapt_params["lora_rank"]
                lora_a = adapt_params["lora_a"]
                lora_drop = adapt_params["lora_drop"]
                self.lora_w1_l1 = nn.Linear(in_features, lora_rank, bias=False)
                self.lora_w1_l2 = nn.Linear(lora_rank, hidden_features, bias=False)
                self.lora_w2_l1 = nn.Linear(hidden_features, lora_rank, bias=False)
                self.lora_w2_l2 = nn.Linear(lora_rank, out_features, bias=False)
                self.lora_scaling = lora_a / lora_rank
                nn.init.normal_(self.lora_w1_l1.weight.data, 0, std=0.02)
                nn.init.normal_(self.lora_w2_l1.weight.data, 0, std=0.02)
                
                # nn.init.kaiming_uniform_(self.lora_w1_l1.weight.data, a=math.sqrt(5))
                # nn.init.kaiming_uniform_(self.lora_w2_l1.weight.data, a=math.sqrt(5))

                nn.init.constant_(self.lora_w1_l2.weight.data, 0)
                nn.init.constant_(self.lora_w2_l2.weight.data, 0)
                self.lora_drop = nn.Dropout(lora_drop)

                if "fixed_adapt_style" in adapt_params:
                    self.fixed_adapt_style = adapt_params["fixed_adapt_style"]
                else:
                    self.fixed_adapt_style = False

    def forward(self, x: Tensor) -> Tensor:
        if self.adapt:
            if not self.fixed_adapt_style:
                x = self.fc1(x) + (
                    self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x))) * self.lora_scaling
                )
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x) + (
                    self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(x))) * self.lora_scaling
                )
                x = self.drop(x)
                return x
            else:
                lora_x = (
                    self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x))) * self.lora_scaling
                )
                lora_x = (
                    self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(lora_x))) * self.lora_scaling
                )
                
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = x + lora_x
                x = self.drop(x)
                return x
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
