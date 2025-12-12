import os
import torch
import torch.nn as nn
from datetime import datetime


class Config:
    def __init__(self, args):
        # --- Base setup ---
        self.seed = 42
        self.pass_channel = True
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.CUDA else "cpu")
        self.norm = False

        # --- Logging and workdir ---
        self.print_step = 100
        self.plot_step = 10000
        self.filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.workdir = f"./history/{self.filename}"
        self.homedir = "/home/matthewwang16czap/"
        # self.homedir = "/home/gexin/"
        # self.homedir = "/public/home/sihanwang/"
        self.log = f"{self.workdir}/Log_{self.filename}.log"
        self.samples = f"{self.workdir}/samples"
        self.models = f"{self.workdir}/models"
        self.logger = None

        os.makedirs(self.samples, exist_ok=True)
        os.makedirs(self.models, exist_ok=True)

        # --- Training details ---
        self.normalize = False
        self.learning_rate = 1e-4
        self.alpha_losses = [10, 10, 10, 1 / 255.0]
        self.tot_epoch = 10_000_000

        # --- Model toggles ---
        self.denoise = args.denoise
        self.denoise_training = args.denoise_training

        # --- Dataset & architecture setup ---
        self._setup_dataset(args)

    def _setup_dataset(self, args):
        """Configure dataset and model-specific parameters."""
        if args.trainset == "CIFAR10":
            self._setup_cifar10(args)
        elif args.trainset == "DIV2K":
            self._setup_div2k(args)
        else:
            raise ValueError(f"Unsupported trainset: {args.trainset}")

    def _setup_cifar10(self, args):
        self.save_model_freq = 5
        self.image_dims = (3, 32, 32)
        self.train_data_dir = self.homedir + "datasets/CIFAR10/"
        self.test_data_dir = self.homedir + "datasets/CIFAR10/"
        self.batch_size = 128
        self.downsample = 2
        self.channel_number = int(args.C)

        self.encoder_kwargs = dict(
            model=args.model,
            img_size=(self.image_dims[1], self.image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 256],
            depths=[2, 4],
            num_heads=[4, 8],
            C=self.channel_number,
            window_size=2,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        self.decoder_kwargs = dict(
            model=args.model,
            img_size=(self.image_dims[1], self.image_dims[2]),
            embed_dims=[256, 128],
            depths=[4, 2],
            num_heads=[8, 4],
            C=self.channel_number,
            window_size=2,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )

    def _setup_div2k(self, args):
        self.save_model_freq = 100
        self.image_dims = (3, 256, 256)
        base_path = self.homedir + "datasets/DIV2K/"
        self.train_data_dir = [
            f"{base_path}/clic2020/**",
            f"{base_path}/clic2021/train",
            f"{base_path}/clic2021/valid",
            f"{base_path}/clic2022/val",
            f"{base_path}/DIV2K_train_HR",
            f"{base_path}/DIV2K_valid_HR",
        ]
        self.batch_size = 16
        self.downsample = 4

        # Testset options
        testset_map = {
            "kodak": [self.homedir + "datasets/Kodak/"],
            "CLIC21": [self.homedir + "datasets/clic2021/test/"],
            "ffhq": [self.homedir + "datasets/ffhq/"],
        }
        self.test_data_dir = testset_map.get(args.testset, [])

        # Model-specific setup
        if args.model in ["SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA"]:
            self.channel_number = int(args.C)
        else:
            self.channel_number = None

        size_map = {
            "small": dict(depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10]),
            "base": dict(depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10]),
            "large": dict(depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10]),
        }

        if args.model_size not in size_map:
            raise ValueError(f"Unknown model_size: {args.model_size}")

        self.encoder_kwargs = dict(
            model=args.model,
            img_size=(self.image_dims[1], self.image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 192, 256, 320],
            depths=size_map[args.model_size]["depths"],
            num_heads=size_map[args.model_size]["num_heads"],
            C=self.channel_number,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )

        self.decoder_kwargs = dict(
            model=args.model,
            img_size=(self.image_dims[1], self.image_dims[2]),
            embed_dims=[320, 256, 192, 128],
            depths=size_map[args.model_size]["depths"][::-1],
            num_heads=size_map[args.model_size]["num_heads"][::-1],
            C=self.channel_number,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
