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
    losses = []
    metrics = []
    losses_last_10 = []
    metrics_last_10 = []

    iteration = 0
    while iteration < options.max_iteration :
        data = model.random_crop()
        loss, metric = model.train_step(data)
        iteration += 1
        model.scheduler.step()
        losses.append(loss)
        losses_last_10.append(loss)
        metrics.append(metric)
        metrics_last_10.append(metric)
        if len(losses_last_10) > 10 :
            losses_last_10.pop(0)
            metrics_last_10.pop(0)

        if iteration % options.snapshot_interval == 0 :
            mean_loss = np.mean(losses_last_10)
            mean_metric = np.mean(metrics_last_10)
            message = ""
            message += f"Iteration: {iteration} "
            message += f"Loss: {loss:.4f}, Mean Loss: {mean_loss:.4f} "
            message += f"Metric: {metric:.4f}, Mean Metric: {mean_metric:.4f}"
            print(message)
            model.save_snapshot(data, iteration)

        # if mean_loss < options.convegence_threshold :
        #     print(f"Converged at iteration {iteration}")
        #     break

    end_time = time.time()
    model.save_networks(iteration, save_latest=True)

    print(f"===== Train Finished =====")
    print(f"Elapsed time: {end_time - start_time:.2f} sec")
    print(f"Total iteration: {iteration}")
    print(f"Final loss: {loss:.4f}")
    print(f"Final metric: {metric:.4f}")

    model.test(options.data_file_path)





    #     model.set_input(data)

    # while iteration < options.max_iteration:

    #     fo

    #     loss = model.train_step()
    #     print(f"Iteration: {iteration}, Loss: {loss:.4f}")
    #     iteration += 1
    #     if loss < options.eps :
    #         break
    # end_time = time.time()

    # print(f"Elapsed time: {end_time - start_time:.2f} sec")


if __name__ == "__main__" :
    train()

