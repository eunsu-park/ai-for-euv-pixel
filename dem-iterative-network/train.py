import time
import numpy as np
from options import Options
from main import DINE
from utils import set_seed


def train():
    options = Options().parse()
    options.phase = "train"

    set_seed(options.seed)

    model = DINE(options)

    start_iteration = 0
    if options.model_path != '' :
        start_iteration = model.load_networks(options.model_path)

    start_time = time.time()
    iteration = start_iteration
    while iteration < options.max_iteration:
        loss = model.train_step()
        print(f"Iteration: {iteration}, Loss: {loss:.4f}")
        iteration += 1
        if loss < options.eps :
            break
    end_time = time.time()


if __name__ == "__main__" :
    train()

