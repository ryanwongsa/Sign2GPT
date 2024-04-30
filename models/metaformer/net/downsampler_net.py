import torch.nn as nn
import torch
from models.metaformer.net.meta_block import basic_blocks
from models.metaformer.downsamplers.downsampler import Downsampler


class Model(nn.Module):
    def __init__(
        self,
        layers,
        embed_dims,
        downsamples,
        mixer_params,
        attention_params,
        mlp_params,
        drop_path_rate=0.1,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        layer_norm_type="pre",
    ):
        super().__init__()
        network = []
        for index in range(len(layers)):
            stage = basic_blocks(
                embed_dims[index],
                index,
                layers,
                mlp_params,
                mixer_params,
                attention_params,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                layer_norm_type=layer_norm_type,
            )
            network.append(stage)
            if index >= len(layers) - 1:
                break
            if downsamples[index] or embed_dims[index] != embed_dims[index + 1]:
                # downsampling between two stages
                network.append(
                    Downsampler(
                        in_dim=embed_dims[index],
                        out_dim=embed_dims[index + 1],
                    )
                )

        self.network = nn.ModuleList(network)

    def forward(self, z, z_mask, return_features=False):
        mask = z_mask
        x = z
        list_of_features = []
        for net in self.network:
            y = net((x, mask))
            x, mask = y
            list_of_features.append((x, mask))
        if return_features:
            return x, mask, list_of_features
        else:
            return x, mask
