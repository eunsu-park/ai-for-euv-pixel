import time
import numpy as np
from options import Options
from main import EPIC
from utils import set_seed


def train():
    options = Options().parse()
    options.phase = "train"

    set_seed(options.seed)

    model = EPIC(options)

    start_epoch = 0
    if options.model_path != '' :
        start_epoch = model.load_networks(options.model_path)

    epoch = start_epoch
    while epoch < options.n_epochs :
        epoch_start_time = time.time()
        t0 = time.time()
        losses = []
        metrics = []
        for i, data_dict in enumerate(model.dataloader) :
            data = data_dict["data"]
            loss, metric = model.train_step(data)
            losses.append(loss)
            metrics.append(metric)

            if (i+1) % options.report_freq == 0 :
                message = ""
                message += f"Epoch [{epoch}/{options.n_epochs}] "
                message += f"Batch [{i+1}/{len(model.dataloader)}] "
                message += f"Loss: {loss:.4f} "
                message += f"Metric: {metric:.4f} "
                message += f"Time: {time.time() - t0:.2f} sec"
                print(message)
                model.save_snapshot(data, epoch, i+1)
                t0 = time.time()

        epoch += 1
        model.scheduler.step()

        print(f"===== Epoch {epoch} Finished =====")
        print(f"Elapsed Time: {time.time() - epoch_start_time:.2f} sec")
        print(f"Loss: {np.mean(losses):.4f})")
        print(f"Metric: {np.mean(metrics):.4f}")

        if epoch % options.save_freq == 0 :
            model.save_networks(epoch)

    print("Training Done")

if __name__ == "__main__" :
    train()
