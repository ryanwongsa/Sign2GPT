from torch import nn
import importlib
import torch

class Model(nn.Module):
    def __init__(
        self,
        stage1_name,
        stage1_params,
        stage1_ckpt,
        post_name,
        post_params,
        lang_backbone_name,
        llm_name,
        adaptor_params,
        pretext_length,
        freeze=False,
        new_token_length=None,
    ):
        super().__init__()

        mod = importlib.import_module(stage1_name, package=None)
        self.sign_model = mod.Model(**stage1_params)
        if stage1_ckpt is not None:
            self.sign_model.load_state_dict(torch.load(stage1_ckpt,map_location='cpu')['model'])
        self.sign_model.post_model = None

        if freeze:
            for name, param in self.sign_model.named_parameters():
                param.requires_grad = False

        lang_mod = importlib.import_module(lang_backbone_name, package=None)
        self.lang_model = lang_mod.XGLMForCausalLM.from_pretrained(llm_name)
        if new_token_length is not None:
            self.lang_model.resize_token_embeddings(new_token_length)
        for name, param in self.lang_model.named_parameters():
            param.requires_grad = False
        self.lang_model.init_adaptor(**adaptor_params)
        lang_dim = self.lang_model.embed_dim
        sign_dim = self.sign_model.dim

        post_mod = importlib.import_module(post_name, package=None)
        self.post_model = post_mod.Model(**post_params, in_dim=sign_dim, out_dim=lang_dim)
        self.pretext_length = pretext_length


    def forward(
        self,
        text_ids,
        text_mask,
        frame_features,
        frame_mask,
        max_len,
        generate=False,
        gen_params={},
    ):
        dict_sign_output = self.sign_model(frame_features, max_len=max_len)

        post_features, post_mask = (
            dict_sign_output["enc_output"]["post_output"]["x"],
            dict_sign_output["enc_output"]["post_output"]["mask"],
        )

        final_post_dict = self.post_model(post_features, post_mask)

        if not generate:
            output = self.lang_model(
                input_ids=text_ids,
                inputs_adaptors=final_post_dict["x"],
                adaptor_mask=final_post_dict["mask"].bool(),
                attention_mask=text_mask.bool(),
                use_cache=False,
            )
            gates = []
            for name, param in self.lang_model.named_parameters():
                if param.requires_grad:
                    if "adaptor_gate" in name:
                        gates.append(param)
            gates = torch.stack(gates, dim=0)
            return {
                "logits": output.logits[:, self.pretext_length - 1 :],
                "enc_output": dict_sign_output,
                "gates": gates,
            }
        else:
            output_ids = self.lang_model.generate(
                input_ids=text_ids,
                inputs_adaptors=final_post_dict["x"],
                adaptor_mask=final_post_dict["mask"].bool(),
                do_sample=False,
                use_cache=False,
                **gen_params,
            )

            return {
                "output_ids": output_ids[:, self.pretext_length :],
                "enc_output": dict_sign_output,
            }