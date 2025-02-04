from configs.standards.standard_meta_model_zero_config import get_sign_encoder, get_proto_head_params
from models.get_models import get_model

model_name, sign_model_params, dim_model = get_sign_encoder()
post_name, post_params = get_proto_head_params(dim_model)
model_params = {
    "sign_model_name": "models.model_sign_encoder.basic_sign_encoder",
    "sign_model_params": sign_model_params,
    "post_name": post_name,
    "post_params": post_params

}

model = get_model(model_name, model_params)
model.cuda()
model.eval()
import torch

x = [torch.randn(65,3,224,224).cuda(), torch.randn(65,3,224,224).cuda()]

y = model(x, 128)