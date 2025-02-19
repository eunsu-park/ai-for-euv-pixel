import os
from abc import ABC, abstractmethod
import torch
import numpy as np
import asdf
from glob import glob
import matplotlib.pyplot as plt


# def load(file_path):
#     tree = asdf.open(file_path)
#     aia_94 = tree["94"][None, ...]
#     aia_131 = tree["131"][None, ...]
#     aia_171 = tree["171"][None, ...]
#     aia_193 = tree["193"][None, ...]
#     aia_211 = tree["211"][None, ...]
#     aia_335 = tree["335"][None, ...]
#     data = np.concatenate([aia_94, aia_131, aia_171, aia_193, aia_211, aia_335], axis=0)
#     return data


def load(file_path):
    tree = asdf.open(file_path)
    data = tree["data"]
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
    data = torch.from_numpy(data.astype(np.float32))
    return data


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
        pattern = os.path.join(options.data_root, "train", "*.asdf")
        self.file_list = sorted(glob(pattern))
        self.nb_data = len(self.file_list)

    def __len__(self):
        return self.nb_data

    def __getitem__(self, index):
        file_path = self.file_list[index]
        data = load(file_path)
        data = to_tensor(data)
        return data


def define_dataset(options):
    dataset = TrainDataset(options)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=options.batch_size,
        shuffle=options.is_train, num_workers=options.nb_workers)
    return dataset, dataloader


def plot_snapshot(inp, out, save_path):
    ## inp = (batch, channel, height, width)
    ## out = (batch, channel, height, width)
    ## difference between inp and out is the same
    ## 94, 131, 171, 193, 211, 335
    ## image size = 1024 x 1024
    ## grid off

    waves = ["94", "131", "171", "193", "211", "335"]

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for i in range(6):
        axes[0, i].imshow(inp[0, i], cmap="gray", vmin=-1, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(out[0, i], cmap="gray", vmin=-1, vmax=1)
        axes[1, i].axis("off")
        axes[0, i].set_title(waves[i])
        axes[2, i].imshow(inp[0, i] - out[0, i], cmap="gray", vmin=-1, vmax=1)

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_features(features, save_path):
    ## features = (batch, channel, height, width)
    ## batch, 64, 1024, 1024
    ## grid off

    fig, axes = plt.subplots(8, 8, figsize=(18, 18))
    for i in range(64):
        feature = features[0, i]
        # median = np.median(feature)
        # std = np.std(feature)
        # vmin = median - std
        # vmax = median + std
        vmin = -1.
        vmax = 1.
        axes[i//8, i%8].imshow(feature, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i//8, i%8].axis("off")
        axes[i//8, i%8].set_title(f"feature {i}")
    
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_snapshot(inp, out, features, save_path):
    np.savez(save_path, inp=inp, out=out, features=features)


if __name__ == "__main__" :
    import time
    import matplotlib.pyplot as plt
    from options import TrainOptions

    options = TrainOptions().parse()
    dataset, dataloader = define_dataset(options)
    print(len(dataset), len(dataloader))
    time.sleep(1)

    for data in dataloader:
        print(data.shape, data.min(), data.max())

    plt.imshow(data[0,0])
    plt.show()

    # dataloader = get_dataloader(options)
    # for data in enumerate(dataloader):
    #     print(data.shape, data.min(), data.max())
    #     break