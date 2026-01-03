import torchvision
from utils import *
import torch
from loss.image_losses import *
from loss.feature_losses import *
import time


def train_one_epoch(
    epoch,
    global_step,
    net,
    train_loader,
    optimizer,
    logger,
    args,
    config,
    scaler,
):
    is_ddp = hasattr(net, "module")
    net.train()
    optimizer.zero_grad(set_to_none=True)

    # Initialize metrics
    metrics_names = ["elapsed", "losses", "psnrs", "ssims", "msssims", "cbrs", "snrs"]
    metrics = {name: AverageMeter() for name in metrics_names}

    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1

        input, valid = data
        input = input.to(config.device)
        valid = valid.to(config.device)

        # Forward and backward pass
        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            (
                recon_image,
                [restored_feature, pred_noise, noisy_feature, feature],
                [mask, feature_H, feature_W],
                [CBR, SNR, chan_param],
                [mse, psnr, ssim, msssim],
                img_loss,
            ) = net(input, valid)
            img_loss = img_loss / config.accum_steps
        # --- Backward pass ---
        if scaler is not None:
            if is_ddp and (batch_idx % config.accum_steps != config.accum_steps - 1):
                # During accumulation, avoid syncing DDP gradients
                with net.no_sync():
                    scaler.scale(img_loss).backward()
            else:
                scaler.scale(img_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            if is_ddp and (batch_idx % config.accum_steps != config.accum_steps - 1):
                with net.no_sync():
                    img_loss.backward()
            else:
                img_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Update metrics
        metrics["elapsed"].update(time.time() - start_time)
        metrics["losses"].update(img_loss.item())
        metrics["cbrs"].update(CBR)
        metrics["snrs"].update(SNR)
        metrics["psnrs"].update(psnr.item())
        metrics["ssims"].update(ssim.item())
        metrics["msssims"].update(msssim.item())

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
                f"Loss {metrics['losses'].val:.2e}",
                f"CBR {metrics['cbrs'].val:.4f}",
                f"SNR {metrics['snrs'].val:.1f}",
                f"PSNR {metrics['psnrs'].val:.3f}",
                f"SSIM {metrics['ssims'].val:.3f}",
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


def test(net, test_loader, logger, args, config):
    config.isTrain = False
    net.eval()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # --- define metrics in a dict ---
    metric_names = [
        "psnrs",
        "ssims",
        "msssims",
        "snrs",
        "chan_params",
        "cbrs",
    ]

    metrics = {name: 0.0 for name in metric_names}
    counts = 0  # for averaging

    multiple_snr = [int(x) for x in args.multiple_snr.split(",")]
    channel_number = [int(x) for x in args.C.split(",")]

    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_chan_param = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
    results_ssim = np.zeros((len(multiple_snr), len(channel_number)))

    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):
                    input, valid = data

                    if i == 0 and j == 0:
                        save_path = get_path(
                            ".", "recons", f"origin_{rank}_{batch_idx}.png"
                        )
                        torchvision.utils.save_image(input[0], save_path)

                    input = input.to(config.device, non_blocking=True)
                    valid = valid.to(config.device, non_blocking=True)

                    (
                        recon_image,
                        [restored_feature, pred_noise, noisy_feature, feature],
                        [mask, feature_H, feature_W],
                        [CBR, SNR, chan_param],
                        [mse, psnr, ssim, msssim],
                        img_loss,
                    ) = net(input, valid, SNR, rate)

                    # --- save recon images ---
                    if rank == 0:
                        save_path = get_path(
                            ".",
                            "recons",
                            f"recon_{rank}_{batch_idx}_{SNR}_{rate}.png",
                        )
                        torchvision.utils.save_image(recon_image[0], save_path)

                    # --- update metrics ---
                    metrics["psnrs"] += psnr.item() * input.size(0)
                    metrics["ssims"] += ssim.item() * input.size(0)
                    metrics["msssims"] += msssim.item() * input.size(0)
                    metrics["cbrs"] += CBR * input.size(0)
                    metrics["snrs"] += SNR * input.size(0)
                    metrics["chan_params"] += chan_param * input.size(0)
                    counts += input.size(0)

            # --- DDP: reduce metrics across ranks ---
            for key in metrics:
                tensor = torch.tensor(metrics[key], device=config.device)
                if dist.is_initialized():
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                metrics[key] = (tensor / (counts * world_size)).item()  # global average

            # --- store results ---
            results_snr[i, j] = metrics["snrs"]
            results_chan_param[i, j] = metrics["chan_params"]
            results_cbr[i, j] = metrics["cbrs"]
            results_psnr[i, j] = metrics["psnrs"]
            results_ssim[i, j] = metrics["ssims"]
            results_msssim[i, j] = metrics["msssims"]
            # --- clear all metric meters ---
            # At the end of the inner loop
            for key in metrics:
                metrics[key] = 0.0
            counts = 0

    if rank == 0 and logger is not None:
        logger.info("Start Test:")
        logger.info(f"SNR: {results_snr.round(1).tolist()}")
        logger.info(f"SNR (denoised): {results_chan_param.round(2).tolist()}")
        logger.info(f"CBR: {results_cbr.round(4).tolist()}")
        logger.info(f"PSNR: {results_psnr.round(3).tolist()}")
        logger.info(f"SSIM: {results_ssim.round(3).tolist()}")
        logger.info(f"MS-SSIM: {results_msssim.round(3).tolist()}")
        logger.info("Finish Test!")


def train_one_epoch_denoiser(
    epoch,
    global_step,
    net,
    train_loader,
    optimizer,
    logger,
    args,
    config,
    scaler,
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
        "chan_params",
        "psnrs",
        "ssims",
        "msssims",
    ]
    metrics = {name: AverageMeter() for name in metric_names}

    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1

        input, valid = data
        input = input.to(config.device)
        valid = valid.to(config.device)

        # Forward pass
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            (
                recon_image,
                [restored_feature, pred_noise, noisy_feature, feature],
                [mask, feature_H, feature_W],
                [CBR, SNR, chan_param],
                [mse, psnr, ssim, msssim],
                img_loss,
            ) = net(input, valid)
            feature = feature.detach()
            noisy_feature = noisy_feature.detach()
            mask = mask.detach()
            noise = noisy_feature - feature

            # ---------------------- Loss Components ---------------------- #
            # (1) Orthogonal loss: encourage pred_noise ⟂ restored_feature
            orth_loss = model.feature_orthogonal_loss(
                restored_feature, noise, pred_noise, mask
            )

            # (2) MSE between restored_feature and ground-truth feature
            mse_loss = model.feature_mse_loss(restored_feature, feature, mask, noise)

            # (3) Self-consistency: D(feature + pred_noise) ≈ feature
            restored_twice, pred_noise_twice = model.feature_denoiser(
                (feature + pred_noise).detach(), mask, SNR
            )
            # restored_twice, pred_noise_twice = model.feature_denoiser(
            #     (feature + pred_noise).detach(),
            #     mask,
            #     SNR,
            #     feature_H,
            #     feature_W,
            # )
            self_loss = model.feature_mse_loss(
                restored_twice, feature, mask, pred_noise
            )

            # ---------------------- Combine ---------------------- #
            a_1, a_2, a_3, a_4 = config.alpha_losses
            total_loss = (
                a_1 * orth_loss + a_2 * mse_loss + a_3 * self_loss + a_4 * img_loss
            )

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # ---------------------- Metric computation ---------------------- #
        metrics["elapsed"].update(time.time() - start_time)
        metrics["losses"].update(total_loss.item())
        metrics["cbrs"].update(CBR)
        metrics["snrs"].update(SNR)
        metrics["psnrs"].update(psnr.item())
        metrics["ssims"].update(ssim.item())
        metrics["msssims"].update(msssim.item())
        metrics["chan_params"].update(chan_param)

        # ---------------------- Logging ---------------------- #
        if global_step % config.print_step == 0:
            logger.info(
                f"[Epoch {epoch} | Step {global_step}] "
                f"Loss {metrics['losses'].val:.2e} | "
                f"CBR {metrics['cbrs'].val:.4f} | "
                f"SNR {metrics['snrs'].val:.2f} | "
                f"SNR(denoised) {metrics['chan_params'].val:.2f} | "
                f"PSNR {metrics['psnrs'].val:.3f} | "
                f"SSIM {metrics['ssims'].val:.3f} | "
                f"MSSSIM {metrics['msssims'].val:.3f} | "
                f"Orth {orth_loss.item():.4f} | MSE {mse_loss.item():.4f} | "
                f"Recon {img_loss.item():.4f}"
            )

            # Reset metrics after each print interval
            for m in metrics.values():
                m.clear()

    return global_step
