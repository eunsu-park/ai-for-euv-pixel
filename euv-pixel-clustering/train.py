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
        start_time = time.time()
        losses = []
        for i, data_dict in enumerate(model.dataloader) :
            data = data_dict["data"]
            loss = model.train_step(data)
            losses.append(loss)

            if (i+1) % options.report_freq == 0 :
                end_time = time.time()
                print(f"Epoch [{epoch}/{options.n_epochs}] Batch [{i+1}/{len(model.dataloader)}] Loss: {loss:.4f} Time: {end_time - start_time:.2f} sec")
                model.save_snapshot(data, epoch, i+1)
                start_time = time.time

        epoch += 1
        model.scheduler.step()

        print(f"===== 에폭 {epoch} 완료 =====")
        print(f"Time: {time.time() - epoch_start_time:.2f}초")
        print(f"Loss: {np.mean(losses):.4f})")

        if epoch % options.save_freq == 0 :
            model.save_networks(epoch)

if __name__ == "__main__" :
    train()
