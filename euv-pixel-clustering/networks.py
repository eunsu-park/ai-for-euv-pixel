import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_euv_channels, num_latent_features, layer_type="pixel"):
        super(Encoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_latent_features = num_latent_features
        self.layer_type = layer_type
        self.build()
        print(self)
        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def build(self):
        if self.layer_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.layer_type == "conv":
            kernel_size, stride, padding = 3, 1, 1
        model = []
        model += [nn.Conv2d(self.num_euv_channels, 1024, kernel_size, stride, padding), nn.SiLU()]
        model += [nn.Conv2d(1024, self.num_latent_features, kernel_size, stride, padding)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, num_euv_channels, num_latent_features, layer_type="pixel"):
        super(Decoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_latent_features = num_latent_features
        self.layer_type = layer_type
        self.build()
        print(self)
        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def build(self):
        if self.layer_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.layer_type == "convolution":
            kernel_size, stride, padding = 3, 1, 1
        model = []
        model += [nn.Conv2d(self.num_latent_features, 1024, kernel_size, stride, padding), nn.SiLU()]
        model += [nn.Conv2d(1024, self.num_euv_channels, kernel_size, stride, padding)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, num_euv_channels, num_latent_features, layer_type="pixel"):
        super(AutoEncoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_latent_features = num_latent_features
        self.layer_type = layer_type
        self.build()
        print(self)
        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def build(self):
        if self.layer_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.layer_type == "convolution":
            kernel_size, stride, padding = 3, 1, 1

        encoder = []
        encoder += [nn.Conv2d(self.num_euv_channels, 1024, kernel_size, stride, padding), nn.SiLU()]
        encoder += [nn.Conv2d(1024, self.num_latent_features, kernel_size, stride, padding)]
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder += [nn.Conv2d(self.num_latent_features, 1024, kernel_size, stride, padding), nn.SiLU()]
        decoder += [nn.Conv2d(1024, self.num_euv_channels, kernel_size, stride, padding)]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon, latent


class VariationalAutoEncoder(nn.Module):
    def __init__(self, num_euv_channels, num_latent_features, layer_type="pixel"):
        super(VariationalAutoEncoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_latent_features = num_latent_features
        self.layer_type = layer_type

        if self.layer_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.layer_type == "convolution":
            kernel_size, stride, padding = 3, 1, 1

        self.conv1 = nn.Conv2d(self.num_euv_channels, 1024, kernel_size, stride, padding)
        self.conv2_mu = nn.Conv2d(1024, num_latent_features, kernel_size, stride, padding)
        self.conv2_logvar = nn.Conv2d(1024, num_latent_features, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(num_latent_features, 1024, kernel_size, stride, padding)
        self.conv4 = nn.Conv2d(1024, self.num_euv_channels, kernel_size, stride, padding)
        self.act = nn.SiLU()

    def encode(self, x):
        h = self.conv1(x)
        h = self.act(h)
        mu = self.conv2_mu(h)
        logvar = self.conv2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent):
        h = self.conv3(latent)
        h = self.act(h)
        h = self.conv4(h)
        return h
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar)
        x_recon = self.decode(latent)
        return x_recon, latent, mu, logvar


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


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


class Loss(nn.Module):
    def __init__(self, network_type):
        super(Loss, self).__init__()

        if network_type == "autoencoder" :
            self.criterion = nn.MSELoss()
        elif network_type == "variational_autoencoder" :
            self.criterion = vae_loss
        else :
            raise NotImplementedError(f"Network type {network_type} is not implemented")

    def forward(self, y_pred, y):
        return self.criterion(y_pred, y)


class Metric(nn.Module):
    def __init__(self, metric_type):
        super(Metric, self).__init__()

        if metric_type == "mse" :
            self.metric = nn.MSELoss()
        elif metric_type == "mae" :
            self.metric = nn.L1Loss()
        elif metric_type == "log_cosh" :
            self.metric = log_cosh_loss
        else :
            raise NotImplementedError(f"Metric type {metric_type} is not implemented")

    def forward(self, y_pred, y):
        return self.metric(y_pred, y)


if __name__ == "__main__" :

    from options import Options
    options = Options().parse()

    num_euv_channels = options.num_euv_channels
    num_latent_features = options.num_latent_features
    network_type = options.network_type
    layer_type = options.layer_type

    if network_type == "autoencoder":
        model = AutoEncoder(num_euv_channels, num_latent_features, layer_type)
    elif network_type == "variable_autoencoder":
        model = VariationalAutoEncoder(num_euv_channels, num_latent_features, layer_type)
    

    inp = torch.randn(1, num_euv_channels, 1024, 1024)
    out = model(inp)
    for o in out:
        print(o.shape)