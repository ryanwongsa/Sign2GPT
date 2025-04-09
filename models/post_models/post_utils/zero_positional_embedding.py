import torch
import math
class ZeroPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int, max_len=2048, *args, **kwargs):
        super().__init__()
        self.dim_model = dim_model
        self.pos = torch.nn.Parameter(torch.zeros(1,max_len,dim_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        return output + self.pos[:, : output.shape[1]]
