from net.modules import *
import torch


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        input_resolution,
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
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=out_dim,
                    input_resolution=(
                        input_resolution[0] // 2,
                        input_resolution[1] // 2,
                    ),
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

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)

        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)


class SwinJSCC_Encoder(nn.Module):
    def __init__(
        self,
        model,
        img_size,
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
        bottleneck_dim=16,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = img_size
        self.H = img_size[0] // (2**self.num_layers)
        self.W = img_size[1] // (2**self.num_layers)
        self.patch_embed = PatchEmbed(img_size, 2, 3, embed_dims[0])
        self.hidden_dim = int(self.embed_dims[len(embed_dims) - 1] * 1.5)
        self.layer_num = layer_num = 7

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                out_dim=int(embed_dims[i_layer]),
                input_resolution=(
                    self.patches_resolution[0] // (2**i_layer),
                    self.patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer != 0 else None,
            )
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        if C != None:
            self.head_list = nn.Linear(embed_dims[-1], C)
        self.apply(self._init_weights)
        # Channel ModNet or Rate ModNet
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(
            nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim)
        )
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[len(embed_dims) - 1]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()
        if model == "SwinJSCC_w/_SAandRA":
            # extra channel modnet
            self.bm_list1 = nn.ModuleList()
            self.sm_list1 = nn.ModuleList()
            self.sm_list1.append(
                nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim)
            )
            for i in range(layer_num):
                if i == layer_num - 1:
                    outdim = self.embed_dims[len(embed_dims) - 1]
                else:
                    outdim = self.hidden_dim
                self.bm_list1.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list1.append(nn.Linear(self.hidden_dim, outdim))
            self.sigmoid1 = nn.Sigmoid()

    def apply_modulation(self, x, cond, sm_list, bm_list, sigmoid, B, H, W):
        HW_scale = H * W // (self.num_layers**4)

        temp = None
        for i in range(self.layer_num):
            temp = sm_list[i](x.detach() if i == 0 else temp)
            bm = bm_list[i](cond).unsqueeze(1).expand(-1, HW_scale, -1)
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
        B, C, H, W = x.size()
        device = x.device

        # backbone
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if model == "SwinJSCC_w/o_SAandRA":
            return self.head_list(x)

        # prepare condition tensors
        snr_batch = (
            torch.tensor(snr, dtype=torch.float, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        rate_batch = (
            torch.tensor(rate, dtype=torch.float, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        if model == "SwinJSCC_w/_SA":
            mod_x, mod = self.apply_modulation(
                x, snr_batch, self.sm_list, self.bm_list, self.sigmoid, B, H, W
            )
            return self.head_list(mod_x)

        if model == "SwinJSCC_w/_RA":
            mod_x, mod = self.apply_modulation(
                x, rate_batch, self.sm_list, self.bm_list, self.sigmoid, B, H, W
            )
            return self.apply_rate_mask(mod_x, mod, rate)

        if model == "SwinJSCC_w/_SAandRA":
            # step 1: SNR
            mod_x, mod_snr = self.apply_modulation(
                x, snr_batch, self.sm_list1, self.bm_list1, self.sigmoid1, B, H, W
            )

            # step 2: RATE
            mod_x, mod_rate = self.apply_modulation(
                mod_x,
                rate_batch,
                self.sm_list,
                self.bm_list,
                self.sigmoid,
                B,
                H,
                W,
            )
            return self.apply_rate_mask(mod_x, mod_rate, rate)

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

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(
                H // (2 ** (i_layer + 1)), W // (2 ** (i_layer + 1))
            )


def create_encoder(**kwargs):
    model = SwinJSCC_Encoder(**kwargs)
    return model


def build_model(config):
    input_image = torch.ones([1, 256, 256], device=config.device)
    model = create_encoder(**config.encoder_kwargs)
    model.to(config.device)
    model(input_image)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10**6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10**9))
