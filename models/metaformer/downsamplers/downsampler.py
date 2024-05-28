import torch
import torch.nn as nn

class Downsampler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
    def forward(self, input):
        x, mask = input
        x = torch.nn.functional.avg_pool1d(
            x.permute(0, 2, 1), 3, stride=2, padding=1
        ).permute(0, 2, 1)
        with torch.no_grad():
            mask = torch.nn.functional.avg_pool1d(
                mask.float(), 3, stride=2, padding=1, count_include_pad=False
            )
            # Ensure that there is atleast something on the encoder side.
            mask[:, 0] = 1.0
        return (x, (mask > 0.0).bool())
