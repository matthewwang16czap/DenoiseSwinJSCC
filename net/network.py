from net.decoder import *
from net.encoder import *
from loss.image_losses import *
from loss.feature_losses import *
from net.channel import Channel
from random import choice
import torch
import torch.nn as nn
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
from net.unet2d import UNet2D
from net.attractor import Attractor
from utils import sample_choice_ddp


class SwinJSCC(nn.Module):
    def __init__(self, args, config):
        super(SwinJSCC, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.feature_mse_loss = FeatureMSELoss()
        self.feature_orthogonal_loss = FeatureOrthogonalLoss(alpha=0.8)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.mse_loss = MSEWithPSNR(normalization=False)
        self.ssim = SSIM(data_range=1.0)
        self.msssim = MS_SSIM(data_range=1.0)
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.channel_number = args.C.split(",")
        for i in range(len(self.channel_number)):
            self.channel_number[i] = int(self.channel_number[i])
        self.downsample = config.downsample
        self.model = args.model
        feature_channels = encoder_kwargs["embed_dims"][-1]
        self.feature_denoiser = (
            Attractor(
                channels=encoder_kwargs["embed_dims"][-1],
                in_depth=3,
                out_depth=3,
            )
            if args.denoise
            else None
        )
        # self.feature_denoiser = (
        #     UNet2D(
        #         channels=encoder_kwargs["embed_dims"][-1],
        #         hidden=encoder_kwargs["embed_dims"][-1],
        #         depth=3,
        #         factor=1,
        #     )
        #     if args.denoise
        #     else None
        # )

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, valid, given_SNR=None, given_rate=None):
        if given_SNR is None:
            # SNR = choice(self.multiple_snr)
            SNR = sample_choice_ddp(self.multiple_snr, input_image.device)
            chan_param = SNR
        else:
            SNR = torch.tensor([given_SNR], device=input_image.device)
            chan_param = SNR

        if given_rate is None:
            # channel_number = choice(self.channel_number)
            channel_number = sample_choice_ddp(self.channel_number, input_image.device)
        else:
            channel_number = torch.tensor([given_rate], device=input_image.device)

        feature, mask, feature_H, feature_W = self.encoder(
            input_image, chan_param, channel_number, self.model
        )
        if self.model == "SwinJSCC_w/o_SAandRA" or self.model == "SwinJSCC_w/_SA":
            CBR = feature.numel() / 2 / input_image.numel()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

        elif self.model == "SwinJSCC_w/_RA" or self.model == "SwinJSCC_w/_SAandRA":
            CBR = channel_number.float() / (2 * 3 * 2 ** (self.downsample * 2))
            with torch.amp.autocast(device_type="cuda", enabled=False):
                avg_pwr = (feature.float().pow(2).sum()) / mask.float().sum().clamp(
                    min=1e-6
                )
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param, avg_pwr)
            else:
                noisy_feature = feature
        noisy_feature = noisy_feature * mask

        # --- Pass noisy feature through feature_denoiser network ---
        if self.feature_denoiser:
            restored_feature, pred_noise = self.feature_denoiser(
                noisy_feature, mask, SNR
            )  # predict noise
            # restored_feature, pred_noise = self.feature_denoiser(
            #     noisy_feature, mask, SNR, feature_H, feature_W
            # )  # predict noise
            # repredict chan_param
            with torch.amp.autocast(device_type="cuda", enabled=False):
                # ---- signal power ----
                signal_power = (feature.float() * mask.float()).pow(
                    2
                ).sum() / mask.float().sum().clamp(min=1e-6)

                # ---- restore MSE ----
                restore_mse = self.feature_mse_loss(
                    restored_feature.float(),
                    feature.float(),
                    mask.float(),
                )

                # ---- SNR recomputation (log10 in fp32 only) ----
                ratio = signal_power / restore_mse.clamp(min=1e-8)
                chan_param = 10.0 * torch.log10(ratio).detach()
        else:
            pred_noise = torch.zeros_like(noisy_feature)
            restored_feature = noisy_feature

        recon_image = self.decoder(
            restored_feature, chan_param, self.model, feature_H, feature_W, valid
        )

        # --- Compute loss and metrics ---
        img_loss, mse, psnr = self.mse_loss(recon_image, input_image, valid)
        img_loss = img_loss.mean()
        # rescale to [0,255] loss to avoid too small loss
        img_loss = img_loss * 255 * 255
        mse = mse.mean()
        psnr = psnr.mean()
        ssim = self.ssim(recon_image, input_image).mean().detach()
        msssim = self.msssim(recon_image, input_image).mean().detach()

        return (
            recon_image,
            [restored_feature, pred_noise, noisy_feature, feature],
            [mask, feature_H, feature_W],
            [
                CBR.item() if isinstance(CBR, torch.Tensor) else CBR,
                SNR.item() if isinstance(SNR, torch.Tensor) else SNR,
                chan_param,
            ],
            [mse, psnr, ssim, msssim],
            img_loss,
        )
