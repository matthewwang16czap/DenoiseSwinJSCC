import torch.optim as optim
from net.network import SwinJSCC
from data.datasets import get_loader
from utils import *

# torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
from loss.distortion import *
from config import Config
from training import *

## Main issue: network fix input image size
if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="SwinJSCC")
    parser.add_argument("--training", action="store_true", help="training or testing")
    parser.add_argument(
        "--denoise-training", action="store_true", help="train the denoiser"
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

    # get configuration
    config = Config(args)

    if args.trainset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1.0, levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1.0, levels=4, channel=3).cuda()

    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = SwinJSCC(args, config)
    # model_path = "./checkpoints/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"
    # model_path = "./checkpoints/pretrained.model"
    # model_path = "history/2025-09-28 18:57:35/models/2025-09-28 18:57:35_EP50.model"
    # load_weights(net, model_path)
    net = net.cuda()

    model_params = [{"params": net.parameters(), "lr": config.learning_rate}]
    train_loader, test_loader = get_loader(args, config)
    optimizer = optim.Adam(model_params, lr=config.learning_rate)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    if args.training or args.denoise_training:
        for epoch in range(steps_epoch, config.tot_epoch):
            global_step = (
                train_one_epoch(
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
                if args.training
                else train_one_epoch_denoiser(
                    epoch,
                    global_step,
                    net,
                    train_loader,
                    optimizer,
                    logger,
                    args,
                    config,
                )
            )
            if (epoch + 1) % config.save_model_freq == 0:
                save_model(
                    net,
                    save_path=config.models
                    + "/{}_EP{}.model".format(config.filename, epoch + 1),
                )
                test(net, test_loader, CalcuSSIM, logger, args, config)
    else:
        test(net, test_loader, CalcuSSIM, logger, args, config)
