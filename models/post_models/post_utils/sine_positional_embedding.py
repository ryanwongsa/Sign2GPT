import torch
import math
class SinePositionalEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int, max_len=2048, *args, **kwargs):
        super().__init__()
        self.dim_model = dim_model
        seq_len = max_len
        pos = (
            torch.arange(0, seq_len, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, self.dim_model)
        )
        dim = (
            torch.arange(0, self.dim_model, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / self.dim_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        pos = pos.unsqueeze(0)

        self.register_buffer("pos", pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        return output + self.pos.type(output.dtype)[:, : output.shape[1]]
