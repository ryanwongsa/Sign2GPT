import numpy as np


def get_sign_encoder():
    model_name = "models.trial_models.test_pretraining"
    dim_model = 512
    dropout = 0.1
    num_heads = 8
    sign_model_params = {
        "spatial_name": "models.spatial_models.frame_models.dino_adaptor_model",
        "spatial_params": {
            "ckpt_dir": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
            "trainable_names": [],
            "adaptor_layers": list(np.arange(9, 12, 1)),
            "adapt_params": {
                "w_lora": True,
                "w_lora_ff": True,
                "lora_rank": 4,
                "lora_drop": 0.1,
                "lora_a": 4.0,
                "rng_init": False,
            },
        },
        "encoder_name": "models.metaformer.meta_model",
        "encoder_params": {
            "emb_name": "models.metaformer.emb.sine_pos",
            "emb_params": {
                "in_dim": dim_model,
                "d_model": dim_model,
                "pos_config": {"name": "my_sine", "dim_model": dim_model},
            },
            "net_name": "models.metaformer.net.downsampler_net",
            "net_params": {
                "drop_path_rate": dropout,
                "use_layer_scale": True,
                "layer_scale_init_value": 1e-5,
                "layer_norm_type": "post",
                "layers": [2, 2],
                "downsamples": [True],
                "embed_dims": [dim_model, dim_model],
                "mixer_params": {
                    "residual_dropout": dropout,
                    "num_heads": num_heads,
                    "use_rotary_embeddings": True,
                },
                "attention_params": {
                    "name": "local_mask",
                    "dropout": dropout,
                    "window_size": 7,
                },
                "mlp_params": {
                    "name": "MLP",
                    "hidden_layer_multiplier": 4,
                    "activation": "gelu",
                    "dropout": dropout,
                },
            },
            "post_name": "models.metaformer.post.identity_head",
            "post_params": {
                "d_model": dim_model,
            },
            "inits": "xavier",
        },
    }
    return model_name, sign_model_params, dim_model

def get_proto_head_params():
    post_params = {
        "in_dim": dim_model,
        "hidden_dim": 300,
        "num_classes": 2533,
        "dropout": 0.2,
        "class_temperature": 0.1,
        "time_temperature": 0.1,
        "dynamic_time_temperatures": False,
        "dynamic_class_temperatures": False,
        "emb_lang": "de",
        "emb_pkl_dir": f"data/phoenix2014t/processed_words.phx_pkl",
        "trainable_emb": True,
    }