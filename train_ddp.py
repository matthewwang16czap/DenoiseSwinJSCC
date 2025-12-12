import os
import argparse
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from net.network import SwinJSCC
from data.datasets_rev import get_loader
from utils import *
from loss.distortion import *
from config import Config
from training import *


def setup_ddp():
    """Initialize distributed environment and return device info."""
    # Get local rank (provided by torchrun)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # Return config dict for convenience
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "device_id": local_rank,
        "device": torch.device(f"cuda:{local_rank}"),
    }


def cleanup_ddp():
    dist.destroy_process_group()


if __name__ == "__main__":
    # --- Parse args ---
    parser = argparse.ArgumentParser(description="SwinJSCC")
    parser.add_argument("--training", action="store_true", help="training or testing")
    parser.add_argument(
        "--denoise-training", action="store_true", help="train the denoiser"
    )
    parser.add_argument(
        "--stage", type=int, default=1, choices=[1, 2, 3], help="denoise training stage"
    )
    parser.add_argument(
        "--trainset",
        type=str,
        default="DIV2K",
        choices=["CIFAR10", "DIV2K"],
        help="train dataset name",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="ffhq",
        choices=["kodak", "CLIC21", "ffhq"],
        help="specify the testset for HR models",
    )
    parser.add_argument(
        "--distortion-metric",
        type=str,
        default="MSE",
        choices=["MSE", "MS-SSIM"],
        help="evaluation metrics",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SwinJSCC_w/_SAandRA",
        choices=[
            "SwinJSCC_w/o_SAandRA",
            "SwinJSCC_w/_SA",
            "SwinJSCC_w/_RA",
            "SwinJSCC_w/_SAandRA",
        ],
        help="SwinJSCC model or SwinJSCC without channel ModNet or rate ModNet",
    )
    parser.add_argument(
        "--channel-type",
        type=str,
        default="awgn",
        choices=["awgn", "rayleigh"],
        help="wireless channel model, awgn or rayleigh",
    )
    parser.add_argument("--C", type=str, default="96", help="bottleneck dimension")
    parser.add_argument(
        "--multiple-snr", type=str, default="10", help="random or fixed snr"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=["small", "base", "large"],
        help="SwinJSCC model size",
    )
    parser.add_argument("--denoise", action="store_true", help="add denoising module")
    args = parser.parse_args()

    ### DDP CHANGE — initialize distributed
    if "LOCAL_RANK" in os.environ:
        ddp_env = setup_ddp()
    else:
        ddp_env = {
            "rank": 0,
            "world_size": 1,
            "device_id": 0,
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        }

    # --- Config and setup ---
    config = Config(args)
    config.device = ddp_env["device"]
    config.device_id = ddp_env["device_id"]

    if args.trainset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1.0, levels=4, channel=3).to(
            config.device
        )
    else:
        CalcuSSIM = MS_SSIM(data_range=1.0, levels=4, channel=3).to(config.device)

    base_seed = 42 + ddp_env["rank"]  # Different seed per GPU
    seed_torch(base_seed)

    logger = logger_configuration(
        config, save_log=(ddp_env["rank"] == 0)
    )  # only rank 0 logs
    if ddp_env["rank"] == 0:
        logger.info(config.__dict__)

    # --- Model ---
    net = SwinJSCC(args, config).to(config.device)
    # model_path = "./checkpoints/pretrained_EP12500.model"
    # model_path = "./checkpoints/fix_snr_fix_cbr_model.model"
    # model_path = "./checkpoints/denoised_EP4600.model"
    # model_path = "./checkpoints/full.model"
    # load_weights(net, model_path)

    ### DDP CHANGE — wrap model
    if ddp_env["world_size"] > 1:
        net = DDP(
            net,
            device_ids=[config.device_id],
            output_device=config.device_id,
            find_unused_parameters=True,
        )

    # Check if model is wrapped in DDP
    is_ddp = hasattr(net, "module")
    model = net.module if is_ddp else net

    # --- Data ---
    train_loader, test_loader, train_sampler = get_loader(
        args, config, rank=ddp_env["rank"], world_size=ddp_env["world_size"]
    )

    # --- Optimizer ---
    # Effective lr
    # lr = (
    #     ddp_env["world_size"] * config.learning_rate
    #     if ddp_env["world_size"] > 1
    #     else config.learning_rate
    # )
    lr = config.learning_rate
    model_params = [{"params": net.parameters(), "lr": lr}]
    optimizer = optim.Adam(model_params, lr=lr)

    global_step = 0
    steps_epoch = global_step // len(train_loader)

    # --- Train or Test ---
    if args.training or args.denoise_training:
        for epoch in range(steps_epoch, config.tot_epoch):

            # debug
            # for name, buf in net.named_buffers():
            #     rank = dist.get_rank() if dist.is_initialized() else 0
            #     print(f"[rank{dist.get_rank()}] buffer {name} shape={tuple(buf.shape)}")
            #     print(
            #         f"[rank{rank}] input_resolution per layer:",
            #         [layer.input_resolution for layer in net.module.encoder.layers],
            #     )
            # if dist.is_initialized():
            #     dist.barrier()

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if args.training:
                global_step = train_one_epoch(
                    epoch,
                    global_step,
                    net,
                    train_loader,
                    optimizer,
                    CalcuSSIM,
                    logger,
                    args,
                    config,
                )
            else:
                # --- Freeze params ---
                for name, param in model.named_parameters():
                    param.requires_grad = False

                if args.stage == 1:
                    for name, param in model.named_parameters():
                        if "feature_denoiser" in name or "adapter" in name:
                            param.requires_grad = True
                elif args.stage == 2:
                    # for name, param in model.decoder.named_parameters():
                    #     if "bm_list" in name or "sm_list" in name or "head_list" in name:
                    #         param.requires_grad = True
                    for name, param in model.named_parameters():
                        if "adapter" in name or "decoder" in name:
                            param.requires_grad = True
                elif args.stage == 3:
                    for name, param in model.named_parameters():
                        # if "encoder" in name or "decoder" in name:
                        #     param.requires_grad = True
                        param.requires_grad = True

                if args.stage == 1:
                    global_step = train_one_epoch_denoiser(
                        epoch,
                        global_step,
                        net,
                        train_loader,
                        optimizer,
                        CalcuSSIM,
                        logger,
                        args,
                        config,
                    )
                else:
                    global_step = train_one_epoch(
                        epoch,
                        global_step,
                        net,
                        train_loader,
                        optimizer,
                        CalcuSSIM,
                        logger,
                        args,
                        config,
                    )

            # Save/check only on rank 0
            if (epoch + 1) % config.save_model_freq == 0 and ddp_env["rank"] == 0:
                save_model(
                    net.module if isinstance(net, DDP) else net,
                    save_path=f"{config.models}/{config.filename}_EP{epoch+1}.model",
                )
                test(
                    net.module if isinstance(net, DDP) else net,
                    test_loader,
                    CalcuSSIM,
                    logger,
                    args,
                    config,
                )
    else:
        test(
            net.module if isinstance(net, DDP) else net,
            test_loader,
            CalcuSSIM,
            logger,
            args,
            config,
        )

    ### DDP CLEANUP
    if ddp_env["world_size"] > 1:
        cleanup_ddp()
