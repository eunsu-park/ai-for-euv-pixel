import random
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


class ToTensor:
    def __init__(self, dtype: torch.dtype=torch.float32):
        self.dtype = dtype
    def __call__(self, data):
        return torch.tensor(data, dtype=self.dtype)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_pattern = f"{data_root}/train/*.h5"
        self.data_list = glob(self.data_pattern)
        self.num_data = len(self.data_list)
        self.read_h5 = ReadH5()
        self.to_tensor = ToTensor()

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        data = self.read_h5(file_path)
        data = self.to_tensor(data)
        return {"data": data, "file_path": file_path}


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_pattern = f"{data_root}/test/*.h5"
        self.data_list = glob(self.data_pattern)
        self.num_data = len(self.data_list)
        self.read_h5 = ReadH5()
        self.to_tensor = ToTensor()

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        data = self.read_h5(file_path)
        data = self.to_tensor(data)
        return {"data": data, "file_path": file_path}


if __name__ == "__main__" :
    from options import Options
    options = Options().parse()

    options.phase = "train"

    dataset = TrainDataset(data_root=options.data_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=options.batch_size,
                                             shuffle=True, num_workers=options.num_workers)

    print(len(dataloader), len(dataloader.dataset))

    for i, data_dict in enumerate(dataloader):
        data = data_dict["data"]
        file_path = data_dict["file_path"]
        print(i, data.shape, data.dtype, data.min(), data.max(), data.device, file_path)
        if i == 100:
            break
