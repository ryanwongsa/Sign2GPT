import torch
import math
class LearnablePositionalEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int, max_len=2048, *args, **kwargs):
        super().__init__()
        self.dim_model = dim_model
        self.position_embeddings = torch.nn.Embedding(max_len, dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = x.unsqueeze(-1) if x.ndim == 2 else x
        positions = torch.arange(x.shape[1], device=x.device)
        
        # Expand positions to match the batch size
        positions = positions.unsqueeze(0).expand(x.shape[0], -1)

        positional_encodings = self.position_embeddings(positions)

        return output + positional_encodings
