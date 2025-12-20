from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def feature_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    noise: Optional[torch.Tensor],
) -> torch.Tensor:

    diff = (pred - target) ** 2

    if mask is not None:
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = denom = torch.tensor(diff.numel(), dtype=diff.dtype, device=diff.device)

    loss = diff.sum() / denom

    if noise is not None:
        noise_norm = (noise**2).sum(dim=(1, 2))
        noise_norm = (noise_norm / denom).sqrt()
        weight = 1.0 / (1.0 + noise_norm)
        loss = loss * weight.mean()

    return loss


class FeatureMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None, noise=None):
        return feature_mse_loss(pred, target, mask, noise)


@torch.jit.script
def feature_orthogonal_loss(
    restored_feature: torch.Tensor,
    noise: torch.Tensor,
    pred_noise: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:

    restored_feature = restored_feature * mask
    noise = noise * mask
    pred_noise = pred_noise * mask

    restored_feature = restored_feature - restored_feature.mean(dim=1, keepdim=True)
    noise = noise - noise.mean(dim=1, keepdim=True)
    pred_noise = pred_noise - pred_noise.mean(dim=1, keepdim=True)

    restored_feature = torch.nn.functional.normalize(restored_feature, dim=2)
    noise = torch.nn.functional.normalize(noise, dim=2)
    pred_noise = torch.nn.functional.normalize(pred_noise, dim=2)

    corr_true = (restored_feature * noise).sum(dim=2)
    corr_pred = (restored_feature * pred_noise).sum(dim=2)

    loss_true = (corr_true**2).mean()
    loss_pred = (corr_pred**2).mean()

    return alpha * loss_true + (1.0 - alpha) * loss_pred


class FeatureOrthogonalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, restored_feature, noise, pred_noise, mask):
        return feature_orthogonal_loss(
            restored_feature, noise, pred_noise, mask, self.alpha
        )


def main():
    B, N, C = 4, 16, 8  # batch, sequence/spatial, channels
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Random predictions and targets
    pred = torch.randn(B, N, C, device=device)
    target = torch.randn(B, N, C, device=device)

    # Random mask: 25% valid
    mask = (torch.rand(B, N, C, device=device) > 0.75).float()

    # Random noise for FeatureMSELoss
    noise = torch.randn(B, N, C, device=device)

    # Initialize losses
    mse_loss_fn = FeatureMSELoss().to(device)
    ortho_loss_fn = FeatureOrthogonalLoss(alpha=0.8).to(device)

    # ---------------------
    # Test FeatureMSELoss
    # ---------------------
    loss_no_mask = mse_loss_fn(pred, target)
    loss_mask = mse_loss_fn(pred, target, mask=mask)
    loss_mask_noise = mse_loss_fn(pred, target, mask=mask, noise=noise)

    print("FeatureMSELoss:")
    print(f"  No mask:       {loss_no_mask.item():.6f}")
    print(f"  With mask:     {loss_mask.item():.6f}")
    print(f"  With mask+noise:{loss_mask_noise.item():.6f}")

    # ---------------------
    # Test FeatureOrthogonalLoss
    # ---------------------
    # Random predicted noise
    pred_noise = torch.randn(B, N, C, device=device)

    ortho_loss_no_mask = ortho_loss_fn(pred, noise, pred_noise, torch.ones_like(mask))
    ortho_loss_mask = ortho_loss_fn(pred, noise, pred_noise, mask)

    print("\nFeatureOrthogonalLoss:")
    print(f"  No mask:   {ortho_loss_no_mask.item():.6f}")
    print(f"  With mask: {ortho_loss_mask.item():.6f}")


if __name__ == "__main__":
    main()
