import os

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from networks import define_networks
from pipeline import define_dataset


class DINE:
    def __init__(self, options):
        self.options = options
        self.device = options.device
        self.C, self.R = define_networks(options)

        self.C.double().to(self.device)
        self.R.double().to(self.device)
        self.init_weights(self.C)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.E.parameters()) + list(self.D.parameters()),
                                    lr=options.lr, betas=(options.beta1, options.beta2))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=options.n_epochs // 4, gamma=0.5)

        self.data = define_dataset(options)

    def init_weights(self, net, init_type='normal', init_gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)
        net.apply(init_func)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_step(self, data):
        self.C.train()
        self.R.train()
        self.optimizer.zero_grad()

        x = data.to(self.device)
        y = self.R(self.C(x))
        loss = self.criterion(y, x)

        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def save_networks(self, epoch):
        save_path = os.path.join(self.options.save_root, "model", f"{epoch}.pth")
        torch.save({"calculator" : self.C.state_dict(),
                    "reconstructor" : self.R.state_dict(),
                    "optimizer" : self.optimizer.state_dict(),
                    "scheduler" : self.scheduler.state_dict(),
                    "epoch" : epoch
                    },
                    save_path)
        print(f"Save model: {save_path}")

    def load_networks(self, epoch):
        load_path = os.path.join(self.options.save_root, "model", f"{epoch}.pth")
        checkpoint = torch.load(load_path)
        self.C.load_state_dict(checkpoint["calculator"])
        self.R.load_state_dict(checkpoint["reconstructor"])
        if self.options.is_train is True:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Load model: {load_path}")
        return checkpoint.get("epoch", 0)

    def save_snapshot(self, data, iteration):
        snap_dir = os.path.join(self.options.save_root, "snapshot")
        self.C.eval()
        self.R.eval()
        with torch.no_grad():
            data = data.to(self.device)
            dem = self.C(data)
            recon = self.R(dem)

        self.C.train()
        self.R.train()

        data = data.cpu().detach().numpy()[0]
        dem = dem.cpu().detach().numpy()[0]
        recon = recon.cpu().detach().numpy()[0]

        fig, ax = plt.subplots(2, self.options.num_euv_channels, figsize=(4*self.options.num_euv_channels, 8))

        for i in range(self.options.num_euv_channels):
            ax[0, i].imshow(data[i], cmap="gray", vmin=-1, vmax=1)
            ax[0, i].axis("off")
            ax[0, i].set_title(f"Original {i}")
            ax[1, i].imshow(recon[i], cmap="gray", vmin=-1, vmax=1)
            ax[1, i].axis("off")
            ax[1, i].set_title(f"Reconstruction {i}")
        plt.savefig(f"{snap_dir}/{iteration}.png", dpi=300)
        plt.close()

        save_path = f"{snap_dir}/{iteration}.h5"
        with h5py.File(save_path, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("dem", data=dem)
            f.create_dataset("recon", data=recon)
