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
    return data


def to_tensor(data):
    return torch.tensor(data, dtype=torch.float64)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, waves, max_iterations):
        self.data = read_data(file_path, waves)
        self.data = normalize_euv(self.data)
        self.data = to_tensor(self.data)
        self.num_data = max_iterations

    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        x = np.random.choice(self.data.shape[1] - 256)
        y = np.random.choice(self.data.shape[2] - 256)
        data = self.data[:, x:x+256, y:y+256]
        return data


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, waves):
        self.data = read_data(file_path, waves)
        self.data = normalize_euv(self.data)
        self.data = to_tensor(self.data)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data


if __name__ == "__main__" :
    from options import Options
    options = Options().parse()

    train_dataset = TrainDataset(options.data_file_path, options.waves, options.max_iteration)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(train_loader):
        inp = data.to(options.device).double()
        print(inp.size(), inp.dtype, inp.device, inp.min(), inp.max())
        if i == 10 :
            break

    import matplotlib.pyplot as plt
    img = np.hstack([data[0, i].numpy() for i in range(6)])
    plt.imshow(img, cmap="gray")
    plt.show()
