import os

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from networks import Calculator, Reconstructor, Loss
from pipeline import get_data
from utils import save_options


class DINE:
    def __init__(self, options):
        self.options = options
        self.device = options.device

        self.C = Calculator(options.num_euv_channels,
                            options.num_temperature_bins,
                            options.model_type).to(self.device).double()

        response_file_path = options.response_file_path
        with h5py.File(response_file_path, "r") as h5:
            factor = h5["factor_all_interpol"][:]
        factor = torch.tensor(factor).double()
        self.R = Reconstructor(factor).to(self.device).double()
        self.init_weights(self.C)


        self.criterion = Loss(options.loss_type)
        self.optimizer = optim.Adam(list(self.C.parameters()),
                                    lr=options.lr, betas=(options.beta1, options.beta2))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=options.max_iteration // 4, gamma=0.5)
        self.metric = Loss(options.metric_type)

        self.data = get_data(options.data_file_path, options.waves).to(self.device).double()

        self.experiment_dir = f"{options.save_root}/{options.experiment_name}"
        self.snapshot_dir = f"{self.experiment_dir}/snapshot"
        if not os.path.exists(self.snapshot_dir) :
            os.makedirs(self.snapshot_dir)
        self.model_dir = f"{self.experiment_dir}/model"        
        if not os.path.exists(self.model_dir) :
            os.makedirs(self.model_dir)
        self.test_dir = f"{self.experiment_dir}/test"
        if not os.path.exists(self.test_dir) :
            os.makedirs(self.test_dir)
        # save_options(options, f"{self.experiment_dir}/options.txt")

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
        data = data.to(self.device).double()
        dem = self.C(data)
        recon = self.R(dem)
        loss = self.criterion(recon, data)
        metric = self.metric(recon, data)
        loss.backward()
        self.optimizer.step()
        return loss.item(), metric.item()

    def random_crop(self):
        size = self.options.crop_size
        x = np.random.choice(self.data.shape[-2] - size)
        y = np.random.choice(self.data.shape[-1] - size)
        return self.data[:, :, x:x+size, y:y+size]

    def save_networks(self, iteration, save_latest=True):
        if save_latest is True :
            save_path = f"{self.experiment_dir}/latest.pth"
        else :
            save_path = f"{self.experiment_dir}/{iteration}.pth"
        torch.save({"calculator" : self.C.state_dict(),
                    "reconstructor" : self.R.state_dict(),
                    "optimizer" : self.optimizer.state_dict(),
                    "scheduler" : self.scheduler.state_dict(),
                    "iteration" : iteration
                    },
                    save_path)
        print(f"Save model: {save_path}")

    def load_networks(self, model_path):
        checkpoint = torch.load(model_path)
        self.E.load_state_dict(checkpoint["encoder"])
        self.D.load_state_dict(checkpoint["decoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Load model: {model_path}")
        return checkpoint.get("iteration", 0)

    def save_snapshot(self, data, iteration):
        save_path = f"{self.snapshot_dir}/{iteration:07d}"
        self.C.eval()
        self.R.eval()
        with torch.no_grad():
            data = data.to(self.device)
            dem = self.C(data)
            recon = self.R(dem)

        data = data.cpu().detach().numpy()[0]
        dem = dem.cpu().detach().numpy()[0]
        recon = recon.cpu().detach().numpy()[0]

        fig, ax = plt.subplots(2, self.options.num_euv_channels, figsize=(4*self.options.num_euv_channels, 8))
        for i in range(self.options.num_euv_channels):
            ax[0, i].imshow(data[i], cmap="gray", vmin=-1, vmax=1)
            ax[0, i].axis("off")
            ax[0, i].set_title(f"Original {self.options.waves[i]}")
            ax[1, i].imshow(recon[i], cmap="gray", vmin=-1, vmax=1)
            ax[1, i].axis("off")
            ax[1, i].set_title(f"Reconstruction {self.options.waves[i]}")
        plt.savefig(f"{save_path}.png", dpi=300)
        plt.close()

        with h5py.File(f"{save_path}.h5", "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("dem", data=dem)
            f.create_dataset("recon", data=recon)

    def test(self, file_path):

        self.C.eval()
        self.R.eval()

        data = get_data(self.options.data_file_path, self.options.waves).to(self.device).double()
        file_name = os.path.basename(file_path)
        save_path = f"{self.test_dir}/{file_name}"
        with torch.no_grad():
            dem = self.C(data)
            recon = self.R(dem)

        data = data.cpu().detach().numpy()[0]
        dem = dem.cpu().detach().numpy()[0]
        recon = recon.cpu().detach().numpy()[0]

        fig, ax = plt.subplots(2, self.options.num_euv_channels, figsize=(4*self.options.num_euv_channels, 8))
        for i in range(self.options.num_euv_channels):
            ax[0, i].imshow(data[i], cmap="gray", vmin=-1, vmax=1)
            ax[0, i].axis("off")
            ax[0, i].set_title(f"Original {self.options.waves[i]}")
            ax[1, i].imshow(recon[i], cmap="gray", vmin=-1, vmax=1)
            ax[1, i].axis("off")
            ax[1, i].set_title(f"Reconstruction {self.options.waves[i]}")
        plt.savefig(f"{save_path}.png", dpi=300)
        plt.close()

        from data.functions import denormalize_dem, denormalize_euv
        data = denormalize_euv(data)
        dem = denormalize_dem(dem)
        recon = denormalize_euv(recon)

        with h5py.File(save_path, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("dem", data=dem)
            f.create_dataset("recon", data=recon)

        ratio = []
        for n in range(self.options.num_euv_channels):
            ratio = np.append(ratio, np.nanmean(recon[n])/np.nanmean(data[n]))
            print(f"{self.options.waves[n]}: {ratio[-1]}")

        plt.plot(self.options.waves, ratio, "o-")
        plt.ylabel("Mean ratio")
        plt.savefig(f"{save_path}_ratio.png", dpi=300)
        plt.close()