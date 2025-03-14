import h5py
import torch
import numpy as np
from data.functions import normalize_euv


def read_data(file_path, waves):
    data = []
    with h5py.File(file_path, "r") as f:
        for n in range(len(waves)):
            wave_data = f[str(waves[n])][:]
            wave_data = np.expand_dims(wave_data, axis=0)
            data.append(wave_data)
    data = np.concatenate(data, axis=0)
    data = np.nan_to_num(data, nan=0.0)
    data = np.expand_dims(data, axis=0)
    return data


def to_tensor(data):
    return torch.tensor(data, dtype=torch.float64)


def get_data(file_path, waves) :
    data = read_data(file_path, waves)
    data = normalize_euv(data)
    data = to_tensor(data)
    return data


if __name__ == "__main__" :
    from options import Options
    options = Options().parse()

    data = get_data(options.data_file_path, options.waves)
    print(data.size(), data.dtype, data.device, data.min(), data.max())

    import matplotlib.pyplot as plt
    img = np.hstack([data[0, i].numpy() for i in range(6)])
    plt.imshow(img, cmap="gray")
    plt.show()
