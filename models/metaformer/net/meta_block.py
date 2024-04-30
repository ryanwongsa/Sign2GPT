import torch.nn as nn
import torch
from timm.models.layers import DropPath, trunc_normal_
from xformers.components import build_attention
from models.metaformer.net.multiheaddispatch import MultiHeadDispatch
from xformers.components.feedforward import build_feedforward
import models.metaformer.net.attentions.local_mask_attention


class MetaFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_params,
        mixer_params,
        attention_params,
        norm_layer=nn.LayerNorm,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        layer_norm_type="pre",  #'post'
    ):
        super().__init__()
        self.layer_norm_type = layer_norm_type
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if self.layer_norm_type == "both":
            self.norm1_2 = norm_layer(dim)
            self.norm2_2 = norm_layer(dim)

        self.token_mixer = MultiHeadDispatch(
            **mixer_params,
            dim_model=dim,
            attention=build_attention(
                {
                    **attention_params,
                    "dim_model": dim,
                    "num_heads": mixer_params["num_heads"],
                }
            ),
        )
        mlp_params = dict(mlp_params)
        mlp_params["dim_model"] = dim
        mlp_params["dim_model_out"] = dim
        self.mlp = build_feedforward(mlp_params)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, input):
        x, mask = input
        if self.layer_norm_type == "pre":
            if self.use_layer_scale:
                x = x + self.drop_path(
                    self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                    * self.token_mixer(self.norm1(x), att_mask=mask)
                )
                x = x + self.drop_path(
                    self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                    * self.mlp(self.norm2(x))
                )
            else:
                x = x + self.drop_path(self.token_mixer(self.norm1(x), att_mask=mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            return (x, mask)
        elif self.layer_norm_type == "both":
            if self.use_layer_scale:
                x = self.norm1(
                    x
                    + self.drop_path(
                        self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                        * self.token_mixer(self.norm1_2(x), att_mask=mask)
                    )
                )
                x = self.norm2(
                    x
                    + self.drop_path(
                        self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                        * self.mlp(self.norm2_2(x))
                    )
                )
            else:
                x = self.norm1(
                    x + self.drop_path(self.token_mixer(self.norm1_2(x), att_mask=mask))
                )
                x = self.norm2(x + self.drop_path(self.mlp(self.norm2_2(x))))
            return (x, mask)
        elif self.layer_norm_type == "post":
            if self.use_layer_scale:
                x = self.norm1(
                    x
                    + self.drop_path(
                        self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                        * self.token_mixer(x, att_mask=mask)
                    )
                )
                x = self.norm2(
                    x
                    + self.drop_path(
                        self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x)
                    )
                )
            else:
                # print(self.token_mixer(self.norm1(x), att_mask=mask).std())
                x = self.norm1(x + self.drop_path(self.token_mixer(x, att_mask=mask)))
                x = self.norm2(x + self.drop_path(self.mlp(x)))
            return (x, mask)


def basic_blocks(
    dim,
    index,
    layers,
    mlp_params,
    mixer_params,
    attention_params,
    drop_path_rate=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
    layer_norm_type="pre",
):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            MetaFormerBlock(
                dim,
                mlp_params,
                mixer_params,
                attention_params,
                norm_layer=nn.LayerNorm,
                drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                layer_norm_type=layer_norm_type,
            )
        )
    blocks = nn.Sequential(*blocks)

    return blocks
