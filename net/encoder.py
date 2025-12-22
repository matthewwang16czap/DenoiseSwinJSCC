from net.modules import *
import torch


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
        downsample=None,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.depth = depth

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

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution
        Returns:
            x: transformed features
            H, W: updated resolution
        """
        for blk in self.blocks:
            x = blk(x, H, W)

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, H, W


class SwinJSCC_Encoder(nn.Module):
    def __init__(
        self,
        model,
        patch_size,
        in_chans,
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

        self.model = model
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer if patch_norm else None,
        )

        # Encoder stages
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
                downsample=(PatchMerging if i_layer < self.num_layers - 1 else None),
            )
            self.layers.append(layer)

        self.norm = norm_layer(embed_dims[-1])

        if C is not None:
            self.head_list = nn.Linear(embed_dims[-1], C)

        # -------- JSCC Modulation --------
        self.hidden_dim = int(embed_dims[-1] * 1.5)
        self.adaptive_layer_num = 7

        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(embed_dims[-1], self.hidden_dim))
        for i in range(self.adaptive_layer_num):
            outdim = (
                embed_dims[-1] if i == self.adaptive_layer_num - 1 else self.hidden_dim
            )
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

        if model == "SwinJSCC_w/_SAandRA":
            self.bm_list1 = nn.ModuleList()
            self.sm_list1 = nn.ModuleList()
            self.sm_list1.append(nn.Linear(embed_dims[-1], self.hidden_dim))
            for i in range(self.adaptive_layer_num):
                outdim = (
                    embed_dims[-1]
                    if i == self.adaptive_layer_num - 1
                    else self.hidden_dim
                )
                self.bm_list1.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list1.append(nn.Linear(self.hidden_dim, outdim))
            self.sigmoid1 = nn.Sigmoid()

        self.apply(self._init_weights)

    def apply_modulation(self, x, cond, sm_list, bm_list, sigmoid):
        temp = None
        for i in range(self.adaptive_layer_num):
            temp = sm_list[i](x.detach() if i == 0 else temp)
            bm = bm_list[i](cond).unsqueeze(1)
            temp = temp * bm
        mod = sigmoid(sm_list[-1](temp))
        return x * mod, mod

    def apply_rate_mask(self, x, mod_val, rate):
        B, HW, C = mod_val.shape
        device = x.device
        mod_val = mod_val.to(device)

        mask_vals = mod_val.sum(dim=1)  # (B, C)
        sorted, idx = mask_vals.sort(dim=1, descending=True)  # top-k
        topk = idx[:, :rate]  # (B, rate)

        # flatten index trick
        add = torch.arange(0, B * C, C, device=device).unsqueeze(1).int()
        flat_indices = topk + add

        # build mask
        mask = torch.zeros(mask_vals.size(), device=device).reshape(-1)
        mask[flat_indices.reshape(-1)] = 1
        mask = mask.reshape(B, C)
        mask = mask.unsqueeze(1).expand(-1, HW, -1)
        return x * mask, mask

    def forward(self, x, snr, rate, model):
        B, C, H, W = x.shape
        device = x.device

        # Patch embedding
        x, H, W = self.patch_embed(x)

        # Backbone
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)

        if model == "SwinJSCC_w/o_SAandRA":
            x = self.head_list(x)

        # Condition tensors
        snr_batch = torch.full((B, 1), snr, device=device)
        rate_batch = torch.full((B, 1), rate, device=device)

        mask = torch.ones(x.size(), device=device)

        if model == "SwinJSCC_w/_SA":
            x, _ = self.apply_modulation(
                x, snr_batch, self.sm_list, self.bm_list, self.sigmoid
            )
            x = self.head_list(x)

        if model == "SwinJSCC_w/_RA":
            x, mod = self.apply_modulation(
                x, rate_batch, self.sm_list, self.bm_list, self.sigmoid
            )
            x, mask = self.apply_rate_mask(x, mod, rate)

        if model == "SwinJSCC_w/_SAandRA":
            x, _ = self.apply_modulation(
                x, snr_batch, self.sm_list1, self.bm_list1, self.sigmoid1
            )
            x, mod = self.apply_modulation(
                x, rate_batch, self.sm_list, self.bm_list, self.sigmoid
            )
            x, mask = self.apply_rate_mask(x, mod, rate)

        return x, mask, H, W

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}


def create_encoder(**kwargs):
    model = SwinJSCC_Encoder(**kwargs)
    return model
