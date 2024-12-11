import os
from abc import ABC, abstractmethod
import torch
import numpy as np
import asdf
from glob import glob


def load(file_path):
    tree = asdf.open(file_path)
    aia_94 = tree["94"][None, ...]
    aia_131 = tree["131"][None, ...]
    aia_171 = tree["171"][None, ...]
    aia_193 = tree["193"][None, ...]
    aia_211 = tree["211"][None, ...]
    aia_335 = tree["335"][None, ...]
    data = np.concatenate([aia_94, aia_131, aia_171, aia_193, aia_211, aia_335], axis=0)
    return data


def normalize(data):
    data = np.clip(data, 0, None)
    data = np.log2(data + 1)
    data = data / 7. - 1.
    return data


def denormalize(data):
    data = (data + 1.) * 7.
    data = 2 ** data - 1
    return data


def to_tensor(data):
    return torch.tensor(data).float()


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, options):
        self.options = options

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class TrainDataset(BaseDataset):
    def __init__(self, options):
        super(TrainDataset, self).__init__(options)
        pattern = os.path.join(options.data_root, "*", "*.asdf")
        self.file_list = sorted(glob(pattern))
        self.nb_data = len(self.file_list)

    def __len__(self):
        return self.nb_data

    def __getitem__(self, index):
        file_path = self.file_list[index]
        data = load(file_path)
        data = normalize(data)
        data = to_tensor(data)
        return data


def define_dataset(options):
    dataset = TrainDataset(options)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=options.batch_size,
        shuffle=options.is_train, num_workers=options.nb_workers)
    return dataset, dataloader


if __name__ == "__main__" :
    import time
    import matplotlib.pyplot as plt
    from options import TrainOptions

    options = TrainOptions().parse()
    dataset, dataloader = define_dataset(options)
    print(len(dataset), len(dataloader))
    time.sleep(10)

    for data in dataloader:
        print(data.shape, data.min(), data.max())

    plt.imshow(data[0,0])
    plt.show()

    # dataloader = get_dataloader(options)
    # for data in enumerate(dataloader):
    #     print(data.shape, data.min(), data.max())
    #     break