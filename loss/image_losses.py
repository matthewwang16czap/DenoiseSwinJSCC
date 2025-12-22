from typing import Optional
import torch
import torch.nn.functional as F


@torch.jit.script
def masked_mse(
    X: torch.Tensor,
    Y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    normalization: bool = False,
    eps: float = 1e-8,
):
    """
    Returns per-image MSE. Shape: (B,)
    """
    if normalization:
        X = (X + 1) / 2
        Y = (Y + 1) / 2

    diff = (X - Y) ** 2 * (data_range**2)

    if mask is None:
        # mean over C,H,W only
        return diff.mean(dim=(1, 2, 3))

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)

    num = (diff * mask).sum(dim=(1, 2, 3))
    den = mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    return num / den


class MSEWithPSNR(torch.nn.Module):
    def __init__(self, normalization=False, data_range=1.0, eps=1e-8):
        super().__init__()
        self.normalization = normalization
        self.data_range = data_range
        self.eps = eps

    def forward(self, X, Y, mask=None):
        mse = masked_mse(
            X,
            Y,
            mask,
            data_range=self.data_range,
            normalization=self.normalization,
            eps=self.eps,
        )

        psnr_val = 10.0 * torch.log10((self.data_range**2) / (mse + self.eps))

        return mse, mse.detach(), psnr_val.detach()
