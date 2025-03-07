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
from main import EPIC


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

    model = EPIC(options)

    start_epoch = 0
    if options.model_path != '' :
        start_epoch = model.load_networks(options.model_path)

    epoch = start_epoch
    iteration = 0
    while epoch < options.n_epochs :
        epoch_start_time = time.time()
        losses = []        

        for i, data in enumerate(model.dataloader) :
            loss = model.train_step(data)
            losses.append(loss)
            iteration += 1

            if i+1 % options.report_freq == 0:
                print(f"Epoch [{epoch}/{options.n_epochs}] Batch [{i+1}/{len(model.dataloader)}] Loss: {loss:.4f}")
                model.save_snapshot(data, i+1)

        epoch += 1
        model.scheduler.step()

        print(f"===== 에폭 {epoch} 완료 =====")
        print(f"Time: {time.time() - epoch_start_time:.2f}초")
        print(f"Loss: {np.mean(losses):.4f})")

        if epoch % options.save_freq == 0 :
            model.save_networks(epoch)

if __name__ == "__main__" :
    train()
