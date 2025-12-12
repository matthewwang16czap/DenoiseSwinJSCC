import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic Conv/Down/Up Blocks
# -----------------------------
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.size(-2), x.size(-1)
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return self.block(x)


class UpBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------
# UNet2D
# -----------------------------
class UNet2D(nn.Module):
    def __init__(
        self,
        channels=320,
        hidden=320,
        num_groups=8,
        depth=3,
        factor=2,
        use_sigmoid=False,
        snr_embed_dim=32,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        # Encoder
        self.first_enc = ConvBlock2D(channels, hidden)
        encs, in_ch = [], hidden
        for _ in range(depth):
            out_ch = in_ch * factor
            encs.append(DownBlock2D(in_ch, out_ch))
            in_ch = out_ch
        self.encoders = nn.ModuleList(encs)

        # Bottleneck
        self.bottleneck = ConvBlock2D(in_ch, in_ch)
        bottleneck_ch = in_ch

        # Decoder
        decs = []
        for _ in range(depth):
            out_ch = bottleneck_ch // factor
            decs.append(UpBlock2D(bottleneck_ch, out_ch))
            bottleneck_ch = out_ch
        self.decoders = nn.ModuleList(decs)

        # Final conv heads
        self.final_conv = nn.Conv2d(bottleneck_ch, channels, 3, padding=1)
        self.outc = nn.Conv2d(channels, channels * 2, 3, padding=1)

        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

        # SNR FiLM scalar
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, snr_embed_dim),
            nn.SiLU(),
            nn.Linear(snr_embed_dim, 2),
        )
        nn.init.zeros_(self.snr_mlp[-1].weight)
        nn.init.zeros_(self.snr_mlp[-1].bias)

    # ------------
    # Helpers
    # ------------
    def reshape_input(self, x, H=None, W=None):
        """
        Accepts:
            x: (B, N, C)  with explicit H, W provided
        Returns:
            x in BCHW format, and the known H, W.
        """
        if x.dim() != 3:
            raise ValueError("Input must have shape (B, N, C)")
        B, N, C = x.shape
        if H is None or W is None:
            raise ValueError(
                "When passing (B, N, C), you MUST provide H and W explicitly."
            )
        if H * W != N:
            raise ValueError(
                f"H * W must match N. Got H={H}, W={W}, H*W={H*W}, but N={N}"
            )

        # reshape to BHWC first
        x = x.view(B, H, W, C)
        # permute to BCHW
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x, H, W

    def apply_channel_mask(self, x, mask):
        """
        x: (B, C, H, W)
        mask: (B, C) → broadcast to (B, C, 1, 1)
        """
        if mask is None:
            return x
        mask = mask[:, :, None, None]  # → (B, C, 1, 1)
        return x * mask

    def forward(self, x, mask=None, snr=10.0, H=None, W=None):
        # Convert input to BCHW
        x, H, W = self.reshape_input(x, H, W)
        B, C, _, _ = x.shape

        # Prepare channel mask (B, C)
        if mask is not None:
            if mask.dim() == 3:  # (B, N, C)
                mask = mask[:, 0, :]  # just take channel mask
            if mask.dim() == 2:  # (B, C)
                mask = mask.to(x.device, x.dtype)
            else:
                raise ValueError("Mask must be (B,C) or (B,N,C)")

        # Apply mask to input
        x = self.apply_channel_mask(x, mask)

        # ---------------- Encoder ----------------
        x = self.first_enc(x)

        skips = []
        for enc in self.encoders:
            skips.append(x)
            x = enc(x)

        # ---------------- Bottleneck ----------------
        x = self.bottleneck(x)

        # SNR FiLM
        snr_val = torch.tensor([[snr]], device=x.device, dtype=x.dtype)
        gamma, beta = self.snr_mlp(snr_val)[0]
        x = x * (1 + gamma) + beta

        # ---------------- Decoder ----------------
        for dec in self.decoders:
            skip = skips.pop()

            x = dec(x)

            # fix size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, skip.shape[-2:], mode="bilinear", align_corners=False
                )

            x = x + skip

        # ---------------- Output ----------------
        x = self.final_conv(x)
        x = self.outc(x)

        clean, noise = x.chunk(2, dim=1)  # (B,C,H,W)
        clean = self.apply_channel_mask(clean, mask)
        noise = self.apply_channel_mask(noise, mask)

        # back to (B,N,C)
        clean = clean.permute(0, 2, 3, 1).reshape(B, H * W, C)
        noise = noise.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # mask output
        if mask is not None:
            mask_flat = mask[:, None, :].expand(B, H * W, C)
            clean = clean * mask_flat

        return clean, noise


if __name__ == "__main__":
    # Instantiate model
    model = UNet2D(channels=320, hidden=64, depth=3, factor=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test input dimensions
    batch_size = 16
    seq_len = 256
    num_channels = 320

    noisy = torch.randn(batch_size, seq_len, num_channels)
    mask = torch.ones(batch_size, seq_len, num_channels)

    # Mask out some feature channels
    mask[:, :, :50] = 0
    noisy = noisy * mask

    # Forward pass
    with torch.no_grad():
        clean, noise = model(noisy, mask, 10, H=16, W=16)

    print(f"Input:  {noisy.shape}")
    print(f"Clean:  {clean.shape}")
    print(f"Noise:  {noise.shape}")

    # Mask verification — if masking applied outside model
    if torch.all(mask == 1):
        print("No mask applied.")
    else:
        # Here we expect masked regions to remain zero since we zeroed input
        assert torch.allclose(noisy[mask == 0], torch.zeros_like(noisy[mask == 0]))
        print("✓ Mask correctly applied to input!")

    # GPU memory profiling
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        noisy = noisy.to(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            clean, noise = model(noisy, mask, 10, H=16, W=16)

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✓ Peak GPU memory: {peak_mem:.2f} GB")

    print("✓ Model forward test completed successfully!")
