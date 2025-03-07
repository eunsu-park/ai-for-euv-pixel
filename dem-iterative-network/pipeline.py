from glob import glob
import h5py
import numpy as np
import torch
#from torch.utils.data import Dataset, DataLoader


class ReadH5:
    def __init__(self, key: str="data"):
        self.key = key
    def __call__(self, file_path):
        with h5py.File(file_path, "r") as f:
            data = f[self.key][:]
        return data


class ReadH5:
    def __init__(self, key: str="data"):
        self.key = key
    def __call__(self, file_path):
        with h5py.File(file_path, "r") as f:
            data = f[self.key][:]
        return data


class ToTensor:
    def __init__(self, dtype: torch.dtype=torch.float64):
        self.dtype = dtype
    def __call__(self, data):
        return torch.tensor(data, dtype=self.dtype)


def define_dataset(options):
    data_file_path = options.data_file_path
    data = ReadH5()(data_file_path)
    data = ToTensor()(data)
    data = torch.unsqueeze(data, 0)
    return data


if __name__ == "__main__" :
    from options import TrainOptions
    options = TrainOptions().parse()

    data = define_dataset(options)
    print(data.shape, data.dtype, data.min(), data.max(), data.device)
