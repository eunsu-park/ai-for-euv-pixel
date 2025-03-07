import os

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from networks import Encoder, Decoder
from pipeline import TrainDataset, TestDataset
from utils import save_options


class EPIC:
    def __init__(self, options):
        self.options = options
        self.device = options.device

        self.E = Encoder(num_euv_channels=options.num_euv_channels,
                         num_latent_features=options.num_latent_features,
                         model_type=options.model_type).to(self.device)
        self.D = Decoder(num_euv_channels=options.num_euv_channels,
                         num_latent_features=options.num_latent_features,
                         model_type=options.model_type).to(self.device)
        self.init_weights(self.E, init_type=options.init_type)
        self.init_weights(self.D, init_type=options.init_type)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.E.parameters()) + list(self.D.parameters()),
                                    lr=options.lr, betas=(options.beta1, options.beta2))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=options.n_epochs // 4, gamma=0.5)

        if options.phase == "train" :
            self.dataset = TrainDataset(data_root=options.data_root)
            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                          batch_size=options.batch_size,
                                                          shuffle=True,
                                                          num_workers=options.num_workers)
        elif options.phase == "test" :
            self.dataset = TestDataset(data_root=options.data_root)
            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                          batch_size=options.batch_size,
                                                          shuffle=False,
                                                          num_workers=options.num_workers)

        self.experiment_dir = f"{options.save_root}/{options.experiment_name}"
        self.snapshot_dir = f"{self.experiment_dir}/snapshot"
        self.model_dir = f"{self.experiment_dir}/model"
        self.test_dir = f"{self.experiment_dir}/test"
        if not os.path.exists(self.snapshot_dir) :
            os.makedirs(self.snapshot_dir)
        if not os.path.exists(self.model_dir) :
            os.makedirs(self.model_dir)
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
        self.E.train()
        self.D.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        z = self.E(data)
        recon = self.D(z)
        loss = self.criterion(recon, data)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self):
        self.E.eval()
        self.D.eval()
        losses = []
        with torch.no_grad():
            for i, data_dict in enumerate(self.dataloader):
                data = data_dict["data"].to(self.device)
                file_path = data_dict["file_path"]
                z = self.E(data)
                recon = self.D(z)
                loss = self.criterion(recon, data)

                data = data.cpu().detach().numpy()
                z = z.cpu().detach().numpy()
                recon = recon.cpu().detach().numpy()

                for i in range(len(file_path)) :
                    file_name = os.path.basename(file_path[i])
                    save_path = f"{self.test_dir}/{file_name}.h5"
                    with h5py.File(save_path, "w") as f:
                        f.create_dataset("data", data=data[i])
                        f.create_dataset("z", data=z[i])
                        f.create_dataset("recon", data=recon[i])
                print(f"Test [{i}/{len(self.dataloader)}] Loss: {loss:.4f}")
                losses.append(loss)
        print(f"Average Loss: {np.mean(losses):.4f}")

    def save_networks(self, epoch):
        save_path = os.path.join(self.model_dir, f"{epoch}.pth")
        torch.save({"encoder" : self.E.state_dict(),
                    "decoder" : self.D.state_dict(),
                    "optimizer" : self.optimizer.state_dict(),
                    "scheduler" : self.scheduler.state_dict(),
                    "epoch" : epoch
                    },
                    save_path)
        print(f"Save model: {save_path}")

    def load_networks(self, model_path):
        # load_path = os.path.join(self.model_dir, f"{epoch}.pth")
        checkpoint = torch.load(model_path)
        self.E.load_state_dict(checkpoint["encoder"])
        self.D.load_state_dict(checkpoint["decoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Load model: {model_path}")
        return checkpoint.get("epoch", 0)

    def save_snapshot(self, data, epoch, iteration):
        save_path = f"{self.snapshot_dir}/{epoch:04d}_{iteration:07d}"
        self.E.eval()
        self.D.eval()
        with torch.no_grad():
            data = data.to(self.device)
            z = self.E(data)
            recon = self.D(z)
        data = data.cpu().detach().numpy()[0]
        z = z.cpu().detach().numpy()[0]
        recon = recon.cpu().detach().numpy()[0]

        fig, ax = plt.subplots(2, self.options.num_euv_channels, figsize=(4*self.options.num_euv_channels, 8))
        for i in range(self.options.num_euv_channels):
            ax[0, i].imshow(data[i], cmap="gray", vmin=-1, vmax=1)
            ax[0, i].axis("off")
            ax[0, i].set_title(f"Original {i}")
            ax[1, i].imshow(recon[i], cmap="gray", vmin=-1, vmax=1)
            ax[1, i].axis("off")
            ax[1, i].set_title(f"Reconstruction {i}")
        plt.savefig(f"{save_path}.png", dpi=300)
        plt.close()

        with h5py.File(f"{save_path}.h5", "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("z", data=z)
            f.create_dataset("recon", data=recon)
