import torch
import torch.nn as nn
import torch.nn.functional as F


class Attractor(nn.Module):
    """
    Per-token attractor denoiser (SDRNN-style).
    Operates on x: (B, L, C) and returns cleaned x: (B, L, C).
    """

    def __init__(self, channels, hidden=None, K=15, symmetrize=True):
        super().__init__()
        self.in_dim = channels
        self.n = hidden or int(2 * channels)  # paper suggests ~2x
        self.K = K
        self.symmetrize = symmetrize

        # input projection: maps token channels -> attractor state
        self.w_in = nn.Linear(self.in_dim, self.n, bias=True)
        # recurrent attractor weights (n x n) as a Parameter (we'll apply symmetrize in forward if desired)
        self.W = nn.Parameter(torch.randn(self.n, self.n) * 0.01)
        # optional bias for the recurrent map (paper often uses no bias in W)
        self.b = nn.Parameter(torch.zeros(self.n))
        # output projection: attractor state -> token channels
        self.w_out = nn.Linear(self.n, self.in_dim, bias=True)

        # activation
        self.act = torch.tanh

        # initialization similar to paper (small weights, plus diag copy hints)
        nn.init.normal_(self.w_in.weight, std=0.01)
        nn.init.normal_(self.w_out.weight, std=0.01)
        nn.init.constant_(self.w_in.bias, 0.0)
        nn.init.constant_(self.w_out.bias, 0.0)
        with torch.no_grad():
            # add small identity on w_in/out diagonals for first min dims (paper suggests copy-forward)
            m = min(self.in_dim, self.n)
            self.w_in.weight[:m, :m].add_(torch.eye(m) * 1.0)
            self.w_out.weight[:m, :m].add_(torch.eye(m) * 1.0)

    def forward(self, x, mask=None, tol=None):
        """
        x: (B, L, C)
        mask: (B, L, C) = 1 for valid channels, 0 for zeroed channels
        returns y: (B, L, C)
        """
        B, L, C = x.shape
        if mask is not None:
            x = x * mask

        # project: (B*L, n)
        x_flat = x.contiguous().view(B * L, C)
        c = self.w_in(x_flat)  # cue injected each iteration (B*L, n)

        # init a_0 = 0 (paper sometimes uses a0=0)
        a = torch.zeros_like(c)

        # optionally symmetrize W for stability: W_sym = 0.5*(W + W.T)
        if self.symmetrize:
            W = 0.5 * (self.W + self.W.t())
        else:
            W = self.W

        # iterate K steps (can use convergence check if tol provided)
        prev_out = None
        for k in range(self.K):
            # linear recurrent step: a <- tanh(W @ a + c + b)
            # use F.linear for efficiency: a = tanh(a @ W.T + c + b)
            a = self.act(F.linear(a, W, self.b) + c)
            if tol is not None and k >= 1:
                out = self.w_out(a)
                diff = (out - prev_out).abs().max()
                if diff.item() < tol:
                    break
                prev_out = out

        y_flat = self.w_out(a)  # (B*L, C)
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
