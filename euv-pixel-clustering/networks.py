import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_euv_channels, num_latent_features, model_type="pixel", latent_output_type="sigmoid"):
        super(Encoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_latent_features = num_latent_features
        self.model_type = model_type
        self.latent_output_type = latent_output_type
        self.build()
        print(self)
        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def build(self):
        if self.model_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.model_type == "conv":
            kernel_size, stride, padding = 3, 1, 1
        model = []
        model += [nn.Conv2d(self.num_euv_channels, 1024, kernel_size, stride, padding), nn.SiLU()]
        model += [nn.Conv2d(1024, self.num_latent_features, kernel_size, stride, padding)]
        if self.latent_output_type == "sigmoid":
            model += [nn.Sigmoid()]
        elif self.latent_output_type == "softmax":
            model += [nn.Softmax(dim=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, num_euv_channels, num_latent_features, model_type="pixel"):
        super(Decoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_latent_features = num_latent_features
        self.model_type = model_type
        self.build()
        print(self)
        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def build(self):
        if self.model_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.model_type == "convolution":
            kernel_size, stride, padding = 3, 1, 1
        model = []
        model += [nn.Conv2d(self.num_latent_features, 1024, kernel_size, stride, padding), nn.SiLU()]
        model += [nn.Conv2d(1024, self.num_euv_channels, kernel_size, stride, padding)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


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

    from options import Options
    options = Options().parse()

    E = Encoder(num_euv_channels=options.num_euv_channels,
                num_latent_features=options.num_latent_features,
                model_type=options.model_type).to(options.device)
    D = Decoder(num_euv_channels=options.num_euv_channels,
                num_latent_features=options.num_latent_features,
                model_type=options.model_type).to(options.device)

    inp = torch.randn(options.batch_size,
                      options.num_euv_channels,
                      256, 256).to(options.device)
    print(inp.size())
    out = E(inp)
    print(out.size())
    out = D(out)
    print(out.size())
