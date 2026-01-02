from net.modules import *
import torch
import datetime


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        upsample=None,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        # Swin blocks (resolution-agnostic)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Upsampling layer (optional)
        self.upsample = (
            upsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
            if upsample is not None
            else None
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution
        Returns:
            x: updated features
            H, W: updated resolution
        """

        # Swin blocks (resolution unchanged)
        for blk in self.blocks:
            x = blk(x, H, W)

        # Upsample (resolution doubles)
        if self.upsample is not None:
            x, H, W = self.upsample(x, H, W)

        return x, H, W


class SwinJSCC_Decoder(nn.Module):
    def __init__(
        self,
        model,
        patch_size,
        out_chans,
        embed_dims,
        depths,
        num_heads,
        C,
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        self.patch_unembed = PatchUnembed(
            patch_size,
            out_chans,
            embed_dims[-1],
            norm_layer if patch_norm else None,
        )

        # Build decoder layers (low → high resolution)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dims[i_layer],
                out_dim=(
                    embed_dims[i_layer + 1]
                    if i_layer < self.num_layers - 1
                    else embed_dims[-1]
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                upsample=(
                    PatchReverseMerging if i_layer < self.num_layers - 1 else None
                ),
            )
            self.layers.append(layer)

        # Project channel symbols back to feature dim
        if C is not None:
            self.head_list = nn.Linear(C, embed_dims[0])

        # ===== SNR modulation (decoder side) =====
        self.hidden_dim = int(embed_dims[0] * 1.5)
        self.adaptive_layer_num = 7

        if model != "SwinJSCC_w/_RA":
            self.bm_list = nn.ModuleList()
            self.sm_list = nn.ModuleList()

            self.sm_list.append(nn.Linear(embed_dims[0], self.hidden_dim))
            for i in range(self.adaptive_layer_num):
                outdim = (
                    embed_dims[0]
                    if i == self.adaptive_layer_num - 1
                    else self.hidden_dim
                )
                self.bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list.append(nn.Linear(self.hidden_dim, outdim))

            self.sigmoid = nn.Sigmoid()

        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    # -------------------------------------------------
    # Modulation (resolution-agnostic)
    # -------------------------------------------------
    def apply_modulation(self, x, snr):
        """
        x: (B, L, C)
        snr: scalar or tensor
        """
        B, L, C = x.shape
        device = x.device

        snr_batch = snr.unsqueeze(0).repeat(B, 1)

        temp = None
        for i in range(self.adaptive_layer_num):
            temp = self.sm_list[i](x.detach() if i == 0 else temp)
            bm = self.bm_list[i](snr_batch).unsqueeze(1)
            temp = temp * bm

        mod = self.sigmoid(self.sm_list[-1](temp))
        return x * mod

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(
        self,
        x,
        snr,
        model,
        H,
        W,
        mask=None,
    ):
        """
        x: (B, L, C)  latent symbols
        """

        # Channel → feature
        if model == "SwinJSCC_w/o_SAandRA":
            x = self.head_list(x)

        elif model == "SwinJSCC_w/_SA":
            x = self.head_list(x)
            x = self.apply_modulation(x, snr)

        elif model == "SwinJSCC_w/_RA":
            pass

        elif model == "SwinJSCC_w/_SAandRA":
            x = self.apply_modulation(x, snr)

        # Initial low-resolution grid
        B, L, C = x.shape
        assert H * W == L, "feature size does not match H, W"

        # Decoder backbone
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        # Tokens → image
        x, H, W = self.patch_unembed(x, H, W)

        # Nornalize to [0,1]
        x = self.tanh(x)
        x = 0.5 * (x + 1.0)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            return x * mask
        else:
            return x

    # -------------------------------------------------
    # Init
    # -------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def create_decoder(**kwargs):
    model = SwinJSCC_Decoder(**kwargs)
    return model
