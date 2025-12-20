from typing import Optional
import torch
import torch.nn.functional as F


@torch.jit.script
def masked_mean(x, mask, eps: float = 1e-8):
    if mask is None:
        return x.mean(dim=(1, 2, 3))
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    num = (x * mask).sum(dim=(1, 2, 3))
    den = mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    return num / den


@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    """
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    """
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


@torch.jit.script
def _gaussian_filter(x, window_1d, use_padding: bool):
    """
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    """
    C = x.shape[1]
    # conssistent dtype
    window_1d = window_1d.to(x.dtype)
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(
        out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C
    )
    return out


@torch.jit.script
def ssim(
    X: torch.Tensor,
    Y: torch.Tensor,
    window: torch.Tensor,
    data_range: float,
    use_padding: bool = False,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
):
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_x = _gaussian_filter(X, window, use_padding)
    mu_y = _gaussian_filter(Y, window, use_padding)

    sigma_x = _gaussian_filter(X * X, window, use_padding) - mu_x**2
    sigma_y = _gaussian_filter(Y * Y, window, use_padding) - mu_y**2
    sigma_xy = _gaussian_filter(X * Y, window, use_padding) - mu_x * mu_y

    cs = (2 * sigma_xy + C2) / (sigma_x + sigma_y + C2 + eps)
    cs = F.relu(cs)

    ssim_map = ((2 * mu_x * mu_y + C1) / (mu_x**2 + mu_y**2 + C1 + eps)) * cs

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        m = _gaussian_filter(mask.float(), window[:1], use_padding)
        ssim_val = masked_mean(ssim_map, m)
        cs_val = masked_mean(cs, m)
    else:
        ssim_val = ssim_map.mean(dim=(1, 2, 3))
        cs_val = cs.mean(dim=(1, 2, 3))

    return ssim_val, cs_val


@torch.jit.script
def ms_ssim(
    X: torch.Tensor,
    Y: torch.Tensor,
    window: torch.Tensor,
    data_range: float,
    weights: torch.Tensor,
    use_padding: bool = False,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
):
    weights = weights / weights.sum()
    vals = []

    for i in range(weights.numel()):
        ssim_val, cs = ssim(X, Y, window, data_range, use_padding, mask, eps)
        vals.append(cs if i < weights.numel() - 1 else ssim_val)

        if i < weights.numel() - 1:
            X = F.avg_pool2d(X, 2, ceil_mode=True)
            Y = F.avg_pool2d(Y, 2, ceil_mode=True)
            if mask is not None:
                mask = F.avg_pool2d(mask, 2, ceil_mode=True)

    vals = torch.stack(vals).clamp_min(eps)
    return torch.prod(vals ** weights[:, None], dim=0)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        window_size=11,
        window_sigma=1.5,
        data_range=1.0,
        channel=3,
        use_padding=False,
    ):
        super().__init__()

        self.data_range = data_range
        self.use_padding = use_padding

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer("window", window)

    def forward(self, X, Y, mask=None):
        ssim_val, _ = ssim(
            X,
            Y,
            window=self.window,
            data_range=self.data_range,
            use_padding=self.use_padding,
            mask=mask,
        )
        return ssim_val


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        window_size=11,
        window_sigma=1.5,
        data_range=1.0,
        channel=3,
        use_padding=False,
        weights=None,
        levels=None,
        eps=1e-8,
    ):
        super().__init__()

        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer("window", window)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        weights = torch.tensor(weights, dtype=torch.float)

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()

        self.register_buffer("weights", weights)

    def forward(self, X, Y, mask=None):
        return 1.0 - ms_ssim(
            X,
            Y,
            window=self.window,
            data_range=self.data_range,
            weights=self.weights,
            use_padding=self.use_padding,
            mask=mask,
            eps=self.eps,
        )


@torch.jit.script
def masked_mse(
    X: torch.Tensor,
    Y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 255.0,
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


@torch.jit.script
def psnr(
    X: torch.Tensor,
    Y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 255.0,
    normalization: bool = False,
    eps: float = 1e-8,
):
    mse = masked_mse(X, Y, mask, data_range, normalization, eps)
    return 10.0 * torch.log10((data_range**2) / (mse + eps))


class MSE(torch.nn.Module):
    """
    Computes pixel-space MSE (0–255 scale).

    Inputs are assumed to be in [0, 1] unless normalization=True.
    """

    def __init__(
        self, normalization: bool = False, data_range: float = 255.0, eps: float = 1e-8
    ):
        super().__init__()
        self.normalization = normalization
        self.data_range = data_range
        self.eps = eps

    def forward(self, X, Y, mask=None):
        return masked_mse(
            X,
            Y,
            mask,
            data_range=self.data_range,
            normalization=self.normalization,
            eps=self.eps,
        )


class PSNR(torch.nn.Module):
    def __init__(
        self, normalization: bool = False, data_range: float = 255.0, eps: float = 1e-8
    ):
        super().__init__()
        self.normalization = normalization
        self.data_range = data_range
        self.eps = eps

    def forward(self, X, Y, mask=None):
        return psnr(X, Y, mask, self.data_range, self.normalization, self.eps)


class Distortion(torch.nn.Module):
    def __init__(self, args):
        super(Distortion, self).__init__()

        if args.distortion_metric == "MSE":
            self.dist = MSE(normalization=False)

        elif args.distortion_metric == "PSNR":
            self.dist = PSNR(normalization=False)

        elif args.distortion_metric == "SSIM":
            self.dist = SSIM()

        elif args.distortion_metric == "MS-SSIM":
            self.dist = MS_SSIM(data_range=1.0, levels=4, channel=3)

        else:
            args.logger.info("Unknown distortion type!")
            raise ValueError

    def forward(self, X, Y, mask=None):
        return self.dist(X, Y, mask).mean()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N, C, H, W = 4, 3, 256, 128

    # Random images in [0, 1]
    img1 = torch.rand(N, C, H, W, device=device)
    img2 = torch.rand(N, C, H, W, device=device)

    # Binary mask: keep center region only
    mask = torch.zeros(N, 1, H, W, device=device)
    mask[:, :, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0

    print("Mask valid ratio:", mask.mean().item())

    # -------------------------
    # MSE
    # -------------------------
    mse = MSE(normalization=False).to(device)

    mse_nomask = mse(img1, img2).mean()
    mse_mask = mse(img1, img2, mask).mean()

    print("MSE (no mask): ", mse_nomask.item())
    print("MSE (mask):    ", mse_mask.item())

    # -------------------------
    # PSNR
    # -------------------------
    psnr_metric = PSNR(normalization=False).to(device)

    psnr_nomask = psnr_metric(img1, img2).mean()
    psnr_mask = psnr_metric(img1, img2, mask).mean()

    print("PSNR (no mask): ", psnr_nomask.item())
    print("PSNR (mask):    ", psnr_mask.item())

    # -------------------------
    # SSIM
    # -------------------------
    ssim_loss = SSIM(data_range=1.0, channel=3).to(device)

    ssim_nomask = ssim_loss(img1, img2).mean()
    ssim_mask = ssim_loss(img1, img2, mask).mean()

    print("SSIM (no mask): ", ssim_nomask.item())
    print("SSIM (mask):    ", ssim_mask.item())

    # -------------------------
    # MS-SSIM
    # -------------------------
    ms_ssim_loss = MS_SSIM(
        data_range=1.0,
        levels=4,
        channel=3,
    ).to(device)

    ms_nomask = ms_ssim_loss(img1, img2).mean()
    ms_mask = ms_ssim_loss(img1, img2, mask).mean()

    print("MS-SSIM (no mask): ", ms_nomask.item())
    print("MS-SSIM (mask):    ", ms_mask.item())
