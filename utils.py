import numpy as np
import math
import torch
import random
import os
import logging
import time

import torch.distributed as dist


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def logger_configuration(config, save_log=False, test_mode=False):
    logger = logging.getLogger("Deep joint source channel coder")

    # Avoid re-adding handlers in case of multiple imports / repeated calls
    if logger.hasHandlers():
        logger.handlers.clear()

    if test_mode:
        config.workdir += "_test"
    if save_log and is_main_process():
        os.makedirs(config.workdir, exist_ok=True)
        os.makedirs(config.samples, exist_ok=True)
        os.makedirs(config.models, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s] %(message)s")

    # --- Only rank 0 prints to stdout ---
    if is_main_process():
        stdhandler = logging.StreamHandler()
        stdhandler.setLevel(logging.INFO)
        stdhandler.setFormatter(formatter)
        logger.addHandler(stdhandler)

    # --- All ranks can log to file if you want shared logs ---
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)

    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger


def makedirs(directory):
    os.makedirs(os.path.dirname(directory), exist_ok=True)


def save_model(model, save_path):
    makedirs(save_path)
    torch.save(model.state_dict(), save_path)


def load_weights(net, model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=False)
    del pretrained


def move_to_cpu(*tensors):
    for t in tensors:
        try:
            if torch.is_tensor(t):
                t.data = t.cpu()
        except Exception:
            pass


def mem_report(tag):
    torch.cuda.synchronize()
    print(
        f"{tag} allocated: {torch.cuda.memory_allocated()/1e6:.1f}MB, reserved: {torch.cuda.memory_reserved()/1e6:.1f}MB"
    )


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
