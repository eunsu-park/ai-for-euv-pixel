import time

import torch

from options import TrainOptions
from utils import fix_seed
from pipeline import define_dataset
from networks import define_network, define_criterion, define_optimizer, define_scheduler


if __name__ == "__main__" :


    options = TrainOptions().parse()
    fix_seed(options.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        if options.gpu_id == '-1':
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{options.gpu_id}")
    else :
        device = torch.device("cpu")

    network = define_network(options)
    print(network)
    network.to(device)

    dataset, dataloader = define_dataset(options)
    print(len(dataset))
    print(len(dataloader))

    criterion = define_criterion(options)
    print(criterion)
    criterion.to(device)

    optimizer = define_optimizer(options, network)
    print(optimizer)

    scheduler = define_scheduler(options, optimizer)
    print(scheduler)

    iterations = 0
    epochs = 0
    losses = []
    t0 = time.time()
    epoch_max = options.nb_epochs + options.nb_epochs_decay

    while epochs < epoch_max:
        for data in dataloader:
            inp = data.clone().to(device)
            tar = data.clone().to(device)
            optimizer.zero_grad()
            out = network(inp)
            loss = criterion(out, tar)
            loss.backward()
            optimizer.step()

            iterations += 1
            losses.append(loss.item())

            if iterations % options.logging_freq == 0:
                t1 = time.time()
                print("Epochs: %d/%d, Iterations: %d, Loss: %.4f, Time: %.4f" % (epochs, epoch_max, iterations, loss.item(), t1 - t0))
                t0 = t1

        scheduler.step()
        epochs += 1

        break


