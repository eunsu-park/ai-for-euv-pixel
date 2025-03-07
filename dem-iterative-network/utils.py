import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_options(options, file_path):
    with open(file_path, "w") as f:
        for k, v in options.items():
            f.write(f"{k}: {v}\n")
    print(f"Save options: {file_path}")
