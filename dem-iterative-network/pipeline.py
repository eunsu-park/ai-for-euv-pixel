from glob import glob
import h5py
import numpy as np
import torch
#from torch.utils.data import Dataset, DataLoader


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_pattern = f"{data_root}/triain/*.h5"
        self.data_list = glob(self.data_pattern)
        self.num_data = len(self.data_list)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        pass