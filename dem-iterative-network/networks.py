import torch
import torch.nn as nn
import h5py
from data.functions import denormalize_dem, normalize_euv
import torch.nn.functional as F


class Calculator(nn.Module):
    def __init__(self, num_euv_channels, num_temperature_bins, model_type):
        super(Calculator, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_temperature_bins = num_temperature_bins
        self.model_type = model_type
        self.build()
        print(self)
        print("The number of parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    def build(self):
        if self.model_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.model_type == "conv":
            kernel_size, stride, padding = 3, 1, 1
        model = []
        # model += [nn.Conv2d(self.num_euv_channels, 1024, kernel_size, stride, padding), nn.SiLU()]
        # model += [nn.Conv2d(1024, 512, kernel_size, stride, padding), nn.SiLU()]
        # model += [nn.Conv2d(512, 256, kernel_size, stride, padding), nn.SiLU()]
        # model += [nn.Conv2d(256, 128, kernel_size, stride, padding), nn.SiLU()]
        # model += [nn.Conv2d(128, 64, kernel_size, stride, padding), nn.SiLU()]
        # model += [nn.Conv2d(64, self.num_temperature_bins, kernel_size, stride, padding)]
        model += [nn.Conv2d(self.num_euv_channels, 1024, kernel_size, stride, padding), nn.SiLU()]
        model += [nn.Conv2d(1024, self.num_temperature_bins, kernel_size, stride, padding)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Reconstructor(nn.Module):
    # def __init__(self, response_function, delta_temperature):
    #     super(Reconstructor, self).__init__()
    #     self.register_buffer("response_function", response_function)
    #     self.register_buffer("delta_temperature", delta_temperature)

    def __init__(self, factor):
        super(Reconstructor, self).__init__()
        self.register_buffer("factor", factor)
        print(self.factor.size(), self.factor.dtype, self.factor.device)
        print(self.factor.min(), self.factor.max())

    def forward(self, x):
        # (B, C, H, W) -> (B, H, W, C)
        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        x = denormalize_dem(x)
        x = torch.matmul(x, self.factor)
        # (B, H, W, C) -> (B, C, H, W)
        x = x.transpose(2, 3)
        x = x.transpose(1, 2)
        x = normalize_euv(x)
        return x


def log_cosh_loss(y_pred, y):
    return torch.mean(torch.log(torch.cosh(y_pred - y)))


def regularized_loss(y_pred, y, alpha=0.5, beta=0.5):
    l1 = F.l1_loss(y_pred, y)
    l2 = F.mse_loss(y_pred, y)
    return alpha * l1 + beta * l2


def relative_error(y_pred, y):
    return torch.mean(torch.abs(y_pred - y) / torch.abs(y))


def r2_score(y_pred, y):
    ss_total = torch.sum((y - torch.mean(y)) ** 2)
    ss_residual = torch.sum((y - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


class Loss(nn.Module):
    def __init__(self, loss_type):
        super(Loss, self).__init__()

        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "log_cosh":
            self.criterion = log_cosh_loss
        else:
            raise NotImplementedError(f"Loss function [{loss_type}] is not implemented")

    def forward(self, y_pred, y):
        return self.criterion(y_pred, y)



if __name__ == "__main__" :

    torch.set_default_dtype(torch.float64)
    from pipeline import TrainDataset

    from options import Options
    options = Options().parse()

    C = Calculator(options.num_euv_channels,
                   options.num_temperature_bins,
                   options.model_type).to(options.device).double()
    response_file_path = options.response_file_path
    with h5py.File(response_file_path, "r") as h5:
        factor = h5["factor_all_interpol"][:]
    factor = torch.tensor(factor).double()
    R = Reconstructor(factor).to(options.device).double()

    train_dataset = TrainDataset(options.data_file_path, options.waves, options.max_iteration)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(train_loader):
        inp = data.to(options.device).double()
        print(inp.size(), inp.dtype, inp.device, inp.min(), inp.max())

        out = C(inp)
        print(out.size(), out.dtype, out.device, out.min(), out.max())

        # (B, C, H, W) -> (B, H, W, C)
        # rec = rec.transpose(1, 2)
        # rec = rec.transpose(2, 3)
        # print(rec.size(), rec.dtype, rec.device, rec.min(), rec.max())
        # rec = denormalize_dem(rec)
        # print(rec.size(), rec.dtype, rec.device, rec.min(), rec.max())
        # rec = torch.matmul(rec, factor)
        # print(rec.size(), rec.dtype, rec.device, rec.min(), rec.max())
        # # (B, H, W, C) -> (B, C, H, W)
        # rec = rec.transpose(2, 3)
        # rec = rec.transpose(1, 2)
        # print(rec.size(), rec.dtype, rec.device, rec.min(), rec.max())
        # rec = normalize_euv(rec)
        # print(rec.size(), rec.dtype, rec.device, rec.min(), rec.max())

        rec = R(out)
        print(rec.size(), rec.dtype, rec.device, rec.min(), rec.max())

        break

        if i == 10 :
            break

    import matplotlib.pyplot as plt
    import numpy as np

    img = np.vstack([
        np.hstack([data[0, i].numpy() for i in range(6)]),
        np.hstack([rec.detach().numpy()[0, i] for i in range(6)])])

    plt.imshow(img, cmap="gray")
    plt.show()


    # out = C(inp)
    # print(out.size())

    # print(inp.dtype)
    # print(C.model[0].weight.dtype)

    # factor = R.factor
    # print(f"factor size : {factor.size()}, factor dtype : {factor.dtype}, factor device : {factor.device}")

    # rec = R(out)
    # print(rec.size())
