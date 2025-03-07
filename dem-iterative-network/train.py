import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from options import TrainOptions
from main import DINE


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train():
    options = TrainOptions().parse()

    snap_dir = f"{options.save_root}/snapshot"
    if not os.path.exists(snap_dir) :
        os.makedirs(snap_dir)

    model_dir = f"{options.save_root}/model"
    if not os.path.exists(model_dir) :
        os.makedirs(model_dir)

    set_seed(options.seed)

    model = DINE(options)

    start_epoch = 0
    if options.model_path != '' :
        start_epoch = model.load_networks(options.model_path)


if __name__ == "__main__" :
    train()

