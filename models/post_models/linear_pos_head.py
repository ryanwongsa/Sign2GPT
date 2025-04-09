import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pos_type, pre_pos, in_dim, out_dim):
        super().__init__()
        # self.ln = torch.nn.LayerNorm(d_model)
        self.fc = nn.Linear(in_dim, out_dim)
        self.pos_type = pos_type
        self.pre_pos = pre_pos
        if self.pre_pos:
            dim = in_dim
        else:
            dim = out_dim
        if self.pos_type=="sine":
            from models.post_models.post_utils.sine_positional_embedding import SinePositionalEmbedding
            self.pos_encoder = SinePositionalEmbedding(dim)
        elif self.pos_type=="learnable":
            from models.post_models.post_utils.learnable_positional_embedding import LearnablePositionalEmbedding
            self.pos_encoder = LearnablePositionalEmbedding(dim)
        elif self.pos_type=="zero":
            from models.post_models.post_utils.zero_positional_embedding import ZeroPositionalEmbedding
            self.pos_encoder = ZeroPositionalEmbedding(dim)



    def forward(self, x, mask):
        # x = self.ln(x)
        if self.pre_pos:
            x = self.pos_encoder(x)
        x = self.fc(x)

        if not self.pre_pos:
            x = self.pos_encoder(x)

        return {
            "x":x, 
            "mask": mask
        }
