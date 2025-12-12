import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv1d + GroupNorm + SiLU blocks."""

    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block: halve temporal length, double channels."""

    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        # Pad to even length to avoid size mismatches
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (0, 1))
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block: double temporal length, halve channels."""

    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
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
        assert depth >= 1, "depth must be >= 1"
        self.use_sigmoid = use_sigmoid
        self.depth = depth

        # Encoder
        self.first_enc = ConvBlock(channels, hidden, num_groups)
        enc_layers = []
        in_ch = hidden
        for _ in range(depth):
            out_ch = in_ch * factor
            enc_layers.append(DownBlock(in_ch, out_ch, num_groups))
            in_ch = out_ch
        self.encoders = nn.ModuleList(enc_layers)

        # Bottleneck
        self.bottleneck = ConvBlock(in_ch, in_ch, num_groups)
        bottleneck_ch = in_ch

        # Decoder
        dec_layers = []
        for _ in range(depth):
            out_ch = bottleneck_ch // factor
            dec_layers.append(UpBlock(bottleneck_ch, out_ch, num_groups))
            bottleneck_ch = out_ch
        self.decoders = nn.ModuleList(dec_layers)

        # Final layers
        self.final_conv = nn.Conv1d(bottleneck_ch, channels, 3, padding=1)
        self.outc = nn.Conv1d(channels, channels * 2, 3, padding=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

        if use_sigmoid:
            self.activation = nn.Sigmoid()

        # --- Learnable SNR mapping ---
        # Takes scalar SNR → modulation scalar(s)
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, snr_embed_dim),
            nn.SiLU(),
            nn.Linear(snr_embed_dim, 2),  # [gamma, beta]
        )
        nn.init.zeros_(self.snr_mlp[-1].weight)
        nn.init.zeros_(self.snr_mlp[-1].bias)

    def forward(self, x, mask=None, snr: float = 10.0):
        if mask is not None:
            x = x * mask
        x = x.permute(0, 2, 1)  # [B, C, N]
        skips = []

        # Encoder
        x = self.first_enc(x)
        for enc in self.encoders:
            skips.append(x)
            x = enc(x)

        # Bottleneck
        x = self.bottleneck(x)

        # --- Learnable SNR modulation ---
        snr_tensor = torch.tensor([[snr]], dtype=x.dtype, device=x.device)
        gamma_beta = self.snr_mlp(snr_tensor)  # [1, 2]
        gamma, beta = gamma_beta[0, 0], gamma_beta[0, 1]
        x = x * (1 + gamma) + beta  # FiLM-style modulation

        # Decoder
        for dec in self.decoders:
            skip = skips.pop(-1)
            x = dec(x)
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(
                    x, size=skip.size(-1), mode="linear", align_corners=False
                )
            x = x + skip

        # Output
        x = self.final_conv(x)
        x = self.outc(x)
        if self.use_sigmoid:
            x = self.activation(x)

        clean, noise = x.chunk(2, dim=1)
        clean = clean.permute(0, 2, 1)
        noise = noise.permute(0, 2, 1)

        if mask is not None:
            clean = clean * mask

        return clean, noise
