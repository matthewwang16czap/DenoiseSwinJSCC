from utils import *
import torch
from loss.distortion import *
from loss.denoise import *
import time


def train_one_epoch(
    epoch, global_step, net, train_loader, optimizer, CalcuSSIM, logger, args, config
):
    net.train()
    # Initialize metrics
    metrics_names = ["elapsed", "losses", "psnrs", "msssims", "cbrs", "snrs"]
    metrics = {name: AverageMeter() for name in metrics_names}

    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1

        # Handle different data formats
        input = data[0] if args.trainset == "CIFAR10" else data
        input = input.to(config.device, non_blocking=True)

        # Forward pass
        (
            recon_image,
            restored_feature,
            pred_noise,
            noisy_feature,
            feature,
            mask,
            CBR,
            SNR,
            real_snr,
            chan_param,
            mse,
            loss_G,
        ) = net(input)

        # Backward pass
        optimizer.zero_grad()
        loss_G.backward()
        optimizer.step()

        # Update metrics
        metrics["elapsed"].update(time.time() - start_time)
        metrics["losses"].update(loss_G.item())
        metrics["cbrs"].update(CBR)
        metrics["snrs"].update(SNR)

        if mse.item() > 0:
            psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
            metrics["psnrs"].update(psnr.item())
            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0.0, 1.0)).mean().item()
            metrics["msssims"].update(msssim)

        # Logging
        if global_step % config.print_step == 0:
            process = (
                (global_step % train_loader.__len__())
                / (train_loader.__len__())
                * 100.0
            )

            log_components = [
                f"Epoch {epoch}",
                f"Step [{batch_idx + 1}/{len(train_loader)}={process:.2f}%]",
                f"Time {metrics['elapsed'].val:.3f}",
                f"Loss {metrics['losses'].val:.3f}",
                f"CBR {metrics['cbrs'].val:.4f}",
                f"SNR {metrics['snrs'].val:.1f}",
                f"PSNR {metrics['psnrs'].val:.3f}",
                f"MSSSIM {metrics['msssims'].val:.3f}",
                f"Lr {config.learning_rate}",
            ]

            logger.info(" | ".join(log_components))

            # Reset metrics after logging
            for metric in metrics.values():
                metric.clear()

    # Final reset of metrics
    for metric in metrics.values():
        metric.clear()

    return global_step


def test(net, test_loader, CalcuSSIM, logger, args, config):
    config.isTrain = False
    net.eval()

    # --- define metrics in a dict ---
    metric_names = ["elapsed", "psnr", "msssim", "snr", "chan_param", "cbr"]
    metrics = {name: AverageMeter() for name in metric_names}

    multiple_snr = [int(x) for x in args.multiple_snr.split(",")]
    channel_number = [int(x) for x in args.C.split(",")]

    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_chan_param = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))

    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):

                    start_time = time.time()

                    input = data[0] if args.trainset == "CIFAR10" else data
                    input = input.to(config.device, non_blocking=True)

                    (
                        recon_image,
                        restored_feature,
                        pred_noise,
                        noisy_feature,
                        feature,
                        mask,
                        CBR,
                        SNR_out,
                        real_snr,
                        chan_param,
                        mse,
                        loss_G,
                    ) = net(input, SNR, rate)

                    # torchvision.utils.save_image(recon_image, os.path.join("/home/matthewwang16czap/projects/SwinJSCC/data/", f"recon/{names[0]}"))

                    # --- update metrics ---
                    metrics["elapsed"].update(time.time() - start_time)
                    metrics["cbr"].update(CBR)
                    metrics["snr"].update(SNR_out)
                    metrics["chan_param"].update(chan_param)

                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
                        metrics["psnr"].update(psnr.item())

                        msssim = (
                            1 - CalcuSSIM(input, recon_image.clamp(0, 1)).mean().item()
                        )
                        metrics["msssim"].update(msssim)

                    # --- logging ---
                    # log = " | ".join(
                    #     [
                    #         f"Time {metrics['elapsed'].val:.3f}",
                    #         f"CBR {metrics['cbr'].val:.4f}",
                    #         f"SNR {metrics['snr'].val:.1f}",
                    #         f"SNR (denoised) {metrics['chan_param'].val:.2f}",
                    #         f"PSNR {metrics['psnr'].val:.3f}",
                    #         f"MSSSIM {metrics['msssim'].val:.3f}",
                    #         f"Lr {config.learning_rate}",
                    #     ]
                    # )
                    # logger.info(log)

            # --- store results ---
            results_snr[i, j] = metrics["snr"].avg
            results_chan_param[i, j] = metrics["chan_param"].avg
            results_cbr[i, j] = metrics["cbr"].avg
            results_psnr[i, j] = metrics["psnr"].avg
            results_msssim[i, j] = metrics["msssim"].avg

            # --- clear all metric meters ---
            for m in metrics.values():
                m.clear()

    logger.info("Start Test:")
    logger.info(f"SNR: {results_snr.round(1).tolist()}")
    logger.info(f"SNR (denoised): {results_chan_param.round(2).tolist()}")
    logger.info(f"CBR: {results_cbr.round(4).tolist()}")
    logger.info(f"PSNR: {results_psnr.round(3).tolist()}")
    logger.info(f"MS-SSIM: {results_msssim.round(3).tolist()}")
    logger.info("Finish Test!")


def train_one_epoch_denoiser(
    epoch, global_step, net, train_loader, optimizer, CalcuSSIM, logger, args, config
):
    net.train()

    # Check if model is wrapped in DDP
    is_ddp = hasattr(net, "module")
    model = net.module if is_ddp else net

    # Initialize metric meters
    metric_names = [
        "elapsed",
        "losses",
        "cbrs",
        "snrs",
        "real_snr",
        "chan_param",
        "psnr_recon",
        "msssim_recon",
        "psnr_no_noise",
        "msssim_no_noise",
    ]
    metrics = {name: AverageMeter() for name in metric_names}

    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1

        # Handle different data formats
        input = data[0] if args.trainset == "CIFAR10" else data
        input = input.to(config.device, non_blocking=True)

        # Forward pass
        (
            recon_image,
            restored_feature,
            pred_noise,
            noisy_feature,
            feature,
            mask,
            CBR,
            SNR,
            real_snr,
            chan_param,
            mse,
            loss_G,
        ) = net(input)
        feature = feature.detach()
        noisy_feature = noisy_feature.detach()
        mask = mask.detach()
        noise = noisy_feature - feature

        # ---------------------- Loss Components ---------------------- #
        # (1) Orthogonal loss: encourage pred_noise ⟂ restored_feature
        orth_loss = masked_orthogonal_loss(
            restored_feature, noise, pred_noise, mask, alpha=0.8
        )

        # (2) MSE between restored_feature and ground-truth feature
        mse_loss = masked_mse_loss(restored_feature, feature, mask, noise)

        # (3) Self-consistency: D(feature + pred_noise) ≈ feature
        restored_twice, pred_noise_twice = model.feature_denoiser(
            (feature + pred_noise).detach(),
            mask,
        )
        self_loss = masked_mse_loss(restored_twice, feature, mask, pred_noise)

        # ---------------------- Combine ---------------------- #
        a_1, a_2, a_3, a_4 = config.alpha_losses
        total_loss = a_1 * orth_loss + a_2 * mse_loss + a_3 * self_loss + a_4 * loss_G

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ---------------------- Metric computation ---------------------- #
        metrics["elapsed"].update(time.time() - start_time)
        metrics["losses"].update(total_loss.item())
        metrics["cbrs"].update(CBR)
        metrics["snrs"].update(SNR)
        metrics["real_snr"].update(real_snr)
        metrics["chan_param"].update(chan_param)

        # --- PSNR and MSSSIM for recon_image ---
        if mse.item() > 0:
            psnr_recon = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
            metrics["psnr_recon"].update(psnr_recon.item())

            msssim_recon = (
                1 - CalcuSSIM(input, recon_image.clamp(0.0, 1.0)).mean().item()
            )
            metrics["msssim_recon"].update(msssim_recon)

        # ---------------------- Logging ---------------------- #
        if global_step % config.print_step == 0:
            logger.info(
                f"[Epoch {epoch} | Step {global_step}] "
                f"Loss {metrics['losses'].val:.4f} | "
                f"CBR {metrics['cbrs'].val:.4f} | "
                f"SNR {metrics['snrs'].val:.2f} | "
                f"SNR(real) {metrics['real_snr'].val:.2f} | "
                f"SNR(denoised) {metrics['chan_param'].val:.2f} | "
                f"PSNR(recon) {metrics['psnr_recon'].val:.3f} | "
                f"MSSSIM(recon) {metrics['msssim_recon'].val:.3f} | "
                f"Orth {orth_loss.item():.4f} | MSE {mse_loss.item():.4f} | "
                f"Recon {loss_G.item():.4f}"
            )

            # Reset metrics after each print interval
            for m in metrics.values():
                m.clear()

        # Optional but recommended: clear cache every few steps
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    return global_step
