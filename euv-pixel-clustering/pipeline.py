import random
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




    # def __init__(self, data_root, spacecraft, instrument,
    #              in_channels, in_size, out_channels, out_size):

    #     if spacecraft == "ahead" :
    #         if instrument == "hi_1":
    #             pattern_beacon = f"{data_root}/beacon/ahead/img/hi_1/*/*/*s7h1A.fts"
    #             pattern_science = f"{data_root}/science/ahead/img/hi_1/*/*/*s4h1A.fts"
    #         elif instrument == "hi_2":
    #             pattern_beacon = f"{data_root}/beacon/ahead/img/hi_2/*/*/*s7h2A.fts"
    #             pattern_science = f"{data_root}/science/ahead/img/hi_2/*/*/*s4h2A.fts"
    #     elif spacecraft == "behind":
    #         if instrument == "hi_1":
    #             pattern_beacon = f"{data_root}/beacon/behind/img/hi_1/*/*/*s7h1B.fts"
    #             pattern_science = f"{data_root}/science/behind/img/hi_1/*/*/*s4h1B.fts"
    #         elif instrument == "hi_2":
    #             pattern_beacon = f"{data_root}/beacon/behind/img/hi_2/*/*/*s7h2B.fts"
    #             pattern_science = f"{data_root}/science/behind/img/hi_2/*/*/*s4h2B.fts"

    #     self.beacon_files = glob(pattern_beacon)
    #     self.science_files = glob(pattern_science)

    #     self.in_channels = in_channels
    #     self.in_size = in_size
    #     self.out_channels = out_channels
    #     self.out_size = out_size

    #     self.nb_beacon = len(self.beacon_files)
    #     self.nb_science = len(self.science_files)
    #     self.nb_data = min(self.nb_beacon, self.nb_science)

    #     self.read_fits = ReadFits()
    #     self.to_tensor = ToTensor()

    # def __len__(self):
    #     return self.nb_data
    
    # def __getitem__(self, idx):

    #     ok_beacon = False
    #     while not ok_beacon :
    #         try :
    #             beacon = self.read_fits(self.beacon_files[idx])
    #             if beacon.shape != (self.in_channels, self.in_size, self.in_size) :
    #                 idx = random.randint(0, self.nb_data-1)
    #                 continue
    #             ok_beacon = True
    #         except :
    #             idx = random.randint(0, self.nb_data-1)
        
    #     ok_science = False
    #     while not ok_science :
    #         try :
    #             science = self.read_fits(self.science_files[idx])
    #             if science.shape != (self.out_channels, self.out_size, self.out_size) :
    #                 idx = random.randint(0, self.nb_data-1)
    #                 continue
    #             ok_science = True
    #         except :
    #             idx = random.randint(0, self.nb_data-1)

    #     beacon = self.to_tensor(beacon)
    #     science = self.to_tensor(science)

    #     return {"beacon": beacon, "science": science}

