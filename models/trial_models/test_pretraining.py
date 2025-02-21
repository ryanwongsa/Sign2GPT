from torch import nn
import importlib


class Model(nn.Module):
    def __init__(
        self,
        sign_model_name,
        sign_model_params,
        post_name,
        post_params,
    ):
        super().__init__()

        mod = importlib.import_module(sign_model_name, package=None)
        self.sign_model = mod.Model(**sign_model_params)

        post_mod = importlib.import_module(post_name, package=None)
        self.post_model = post_mod.HeadModel(**post_params)
        self.dim = sign_model_params["encoder_params"]["emb_params"]["d_model"]

    def forward(
        self,
        frame_features,
        max_len,
    ):
        dict_sign_output = self.sign_model(frame_features, max_len=max_len)
        post_features, post_mask = (
            dict_sign_output["enc_output"]["post_output"]["x"],
            dict_sign_output["enc_output"]["post_output"]["mask"],
        )
        if self.post_model is not None:
            dict_post_output = self.post_model(post_features, post_mask)
        else:
            dict_post_output = {}

        return {
            "dict_post_output": dict_post_output,
            **dict_sign_output,
        }

