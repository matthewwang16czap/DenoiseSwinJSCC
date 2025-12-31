import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(in_dim, hidden_dim, out_dim, depth, act=nn.ReLU):
    layers = []
    if depth == 1:
        return nn.Linear(in_dim, out_dim)

    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(act())
    for _ in range(depth - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class Attractor(nn.Module):
    def __init__(
        self,
        channels,
        hidden=None,
        K=15,
        symmetrize=True,
        in_depth=1,
        out_depth=1,
        snr_adaptive=False,
        snr_embed_dim=32,
    ):
        super().__init__()

        self.in_dim = channels
        self.n = hidden or int(2 * channels)
        self.K = K
        self.symmetrize = symmetrize
        self.snr_adaptive = snr_adaptive

        # -------- Input / Output MLPs --------
        self.in_proj = make_mlp(
            in_dim=self.in_dim,
            hidden_dim=self.n,
            out_dim=self.n,
            depth=in_depth,
        )

        self.out_proj = make_mlp(
            in_dim=self.n,
            hidden_dim=self.n,
            out_dim=self.in_dim,
            depth=out_depth,
        )

        # -------- Attractor core --------
        self.W = nn.Parameter(torch.randn(self.n, self.n) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.n))

        # -------- Optional SNR conditioning --------
        if self.snr_adaptive:
            self.snr_embed = nn.Sequential(
                nn.Linear(1, snr_embed_dim),
                nn.ReLU(),
                nn.Linear(snr_embed_dim, 2 * self.n),
            )

        self.act = torch.tanh
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

        with torch.no_grad():
            m = min(self.in_dim, self.n)
            if isinstance(self.in_proj, nn.Linear):
                self.in_proj.weight[:m, :m].add_(torch.eye(m))
            if isinstance(self.out_proj, nn.Linear):
                self.out_proj.weight[:m, :m].add_(torch.eye(m))

    def forward(self, x, mask=None, snr=None, tol=None):
        """
        x:   (B, L, C)
        snr: scalar float or 1-element tensor (only if snr_adaptive=True)
        """

        B, L, C = x.shape
        if mask is not None:
            x = x * mask

        x_flat = x.view(B * L, C)
        c = self.in_proj(x_flat)

        # -------- Optional SNR conditioning --------
        if self.snr_adaptive:
            if snr is None:
                raise ValueError("snr must be provided when snr_adaptive=True")

            if not torch.is_tensor(snr):
                snr = torch.tensor([snr], device=x.device, dtype=x.dtype)
            else:
                snr = snr.to(device=x.device, dtype=x.dtype).view(1)

            gamma, beta = self.snr_embed(snr).chunk(2, dim=-1)
            gamma = gamma.expand(B * L, -1)
            beta = beta.expand(B * L, -1)
        else:
            gamma = beta = None

        # -------- Attractor dynamics --------
        a = torch.zeros_like(c)
        W = 0.5 * (self.W + self.W.T) if self.symmetrize else self.W

        prev_out = None
        for k in range(self.K):
            a = F.linear(a, W, self.b) + c

            if self.snr_adaptive:
                a = gamma * a + beta

            a = self.act(a)

            if tol is not None and k >= 1:
                out = self.out_proj(a)
                if (out - prev_out).abs().max() < tol:
                    break
                prev_out = out

        y_flat = self.out_proj(a)
        y = y_flat.view(B, L, C)

        if mask is not None:
            y = y * mask

        return y, x - y


def test_attractor():
    torch.manual_seed(42)

    B, L, C = 4, 5, 6  # batch, sequence length, channels
    x = torch.randn(B, L, C)

    # create random mask: 0 or 1 per channel per token
    mask = (torch.rand(B, L, C) > 0.3).float()  # ~70% chance channel is kept

    attractor = Attractor(channels=C, hidden=12, K=5, symmetrize=True)

    # forward pass
    y, a = attractor(x, mask=mask, return_all=True)

    print("Input x:\n", x)
    print("Mask:\n", mask)
    print("Output y:\n", y)
    print("Attractor states a:\n", a)

    # check masked positions are zero
    masked_zeros = (y[mask == 0] == 0).all().item()
    print("Masked positions zeroed:", masked_zeros)
    assert masked_zeros, "Masked channels should be zero in output!"

    # check shape consistency
    assert y.shape == x.shape, "Output shape mismatch"
    assert a.shape == (B, L, attractor.n), "Attractor state shape mismatch"

    print("Attractor test passed!")


if __name__ == "__main__":
    test_attractor()
