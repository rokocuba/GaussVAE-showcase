import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import yaml


def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def get_latest_ckpt_step(load_path):
    saved_steps = [int(os.path.splitext(path)[0].split("-")[-1]) for path in os.listdir(load_path) if path.endswith(".pt")]
    latest_step = -1 if len(saved_steps) == 0 else max(saved_steps)
    return latest_step


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg(cfg_path: str, parser: ArgumentParser) -> ArgumentParser:
    with open(cfg_path, "r", encoding='utf-8') as file:
        cfg: dict = yaml.safe_load(file)
    for key, value in cfg.items():
        if value is None:
            raise ValueError("'None' is not a supported value in the config file")
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    return parser


def save_cfg(path: str, args, mode="w"):
    with open(path, mode=mode, encoding='utf-8') as file:
        print("#################### Training Config ####################", file=file)
        yaml.dump(vars(args), file, default_flow_style=False)
