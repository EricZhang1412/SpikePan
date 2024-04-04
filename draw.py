import torch
import numpy as np
from einops import rearrange


def linstretch(ImageToView, tol_low, tol_high):
    N, M = ImageToView.shape
    NM = N * M
    b = ImageToView[:, :].reshape(NM, 1).to(torch.float32)
    sorted_b, _ = torch.sort(b, dim=0)
    t_low = sorted_b[int(NM * tol_low)]
    t_high = sorted_b[int(NM * tol_high)]
    b = torch.clamp((b - t_low) / (t_high - t_low), 0, 1)
    return b.reshape(N, M)

def to_rgb(x, tol_low=0.01, tol_high=0.99):
    x = torch.Tensor(x)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() == 3:
        has_batch = False
        x = x.unsqueeze(0)
    else:
        has_batch = True
    # Try to detect BCHW or BHWC
    if x.shape[1] > 8:
        x = rearrange(x, 'b h w c -> b c h w')
    c = x.shape[1]
    if c == 1:
        x = torch.cat([x, x, x], dim=1)
    elif c == 3:
        pass
    elif c == 4:
        x = x[:, [2, 1, 0], :, :]
    elif c == 8:
        x = x[:, [4, 2, 1], :, :]
    else:
        raise ValueError(f"Unsupported channel number: {c}")
    b, c, h, w = x.shape
    x = rearrange(x, 'b c h w -> c (b h w)')
    sorted_x, _ = torch.sort(x, dim=1)
    t_low = sorted_x[:, int(b * h * w * tol_low)].unsqueeze(1)
    t_high = sorted_x[:, int(b * h * w * tol_high)].unsqueeze(1)
    x = torch.clamp((x - t_low) / (t_high - t_low), 0, 1)
    x = rearrange(x, 'c (b h w) -> b h w c', b=b, c=c, h=h, w=w)
    if not has_batch:
        x = x.squeeze(0)
    return x.cpu().numpy()