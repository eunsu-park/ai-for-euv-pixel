import os
import time
import torch
from options import TrainOptions
from utils import fix_seed
from pipeline import define_dataset, plot_snapshot, plot_features, save_snapshot
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

    model_dir = os.path.join(options.save_root, options.mode, "model")
    os.makedirs(model_dir, exist_ok=True)

    snapshot_dir = os.path.join(options.save_root, options.mode, "snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)

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

                inp = data.clone().to(device)

                network.eval()
                with torch.no_grad():
                    out = network(inp)
                    features = network.encoder(inp)
                network.train()


                inp = inp.cpu().detach().numpy()
                out = out.cpu().detach().numpy()
                features = features.cpu().detach().numpy()

                snapshot_path = os.path.join(snapshot_dir, f"{iterations:08d}")
                save_snapshot(inp, out, features, f"{snapshot_path}.npz")
                plot_snapshot(inp, out, f"{snapshot_path}_recon.png")
                plot_features(features, f"{snapshot_path}_features.png")

                losses = []
                t0 = t1

        scheduler.step()
        epochs += 1

        if epochs % options.model_save_freq == 0 :
            model_path = os.path.join(model_dir, f"{epochs:08d}.pth")
            torch.save(network.state_dict(), model_path)
            print(f"Save model: {model_path}")

