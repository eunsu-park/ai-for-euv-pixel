import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_euv_channels, num_hidden_features, num_latent_features, layer_type="pixel"):
        super(Encoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_hidden_features = num_hidden_features
        self.num_latent_features = num_latent_features
        self.layer_type = layer_type
        self.build()

    def build(self):
        if self.layer_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.layer_type == "conv":
            kernel_size, stride, padding = 3, 1, 1
        encoder = []
        encoder += [nn.Conv2d(self.num_euv_channels,       self.num_hidden_features//2, kernel_size, stride, padding), nn.SiLU()]
        encoder += [nn.Conv2d(self.num_hidden_features//2, self.num_hidden_features,    kernel_size, stride, padding), nn.SiLU()]
        encoder += [nn.Conv2d(self.num_hidden_features,    self.num_latent_features,    kernel_size, stride, padding)]
        self.encoder = nn.Sequential(*encoder)
        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, num_euv_channels, num_hidden_features, num_latent_features, layer_type="pixel"):
        super(Decoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_hidden_features = num_hidden_features
        self.num_latent_features = num_latent_features
        self.layer_type = layer_type
        self.build()

    def build(self):
        if self.layer_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.layer_type == "conv":
            kernel_size, stride, padding = 3, 1, 1
        decoder = []
        decoder += [nn.Conv2d(self.num_latent_features,    self.num_hidden_features,    kernel_size, stride, padding), nn.SiLU()]
        decoder += [nn.Conv2d(self.num_hidden_features,    self.num_hidden_features//2, kernel_size, stride, padding), nn.SiLU()]
        decoder += [nn.Conv2d(self.num_hidden_features//2, self.num_euv_channels,       kernel_size, stride, padding)]
        self.decoder = nn.Sequential(*decoder)
        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, num_euv_channels, num_hidden_features, num_latent_features, layer_type="pixel"):
        super(AutoEncoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_hidden_features = num_hidden_features
        self.num_latent_features = num_latent_features
        self.layer_type = layer_type
        self.build()

    def build(self):
        self.encoder = Encoder(self.num_euv_channels, self.num_hidden_features, self.num_latent_features, self.layer_type)
        self.decoder = Decoder(self.num_euv_channels, self.num_hidden_features, self.num_latent_features, self.layer_type)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent), latent


class VAEEncoder(nn.Module):
    def __init__(self, num_euv_channels, num_hidden_features, num_latent_features, layer_type="pixel"):
        super(VAEEncoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_hidden_features = num_hidden_features
        self.num_latent_features = num_latent_features
        self.layer_type = layer_type
        self.build()

    def build(self):
        if self.layer_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.layer_type == "conv":
            kernel_size, stride, padding = 3, 1, 1
        encoder = []
        encoder += [nn.Conv2d(self.num_euv_channels,    self.num_hidden_features, kernel_size, stride, padding), nn.SiLU()]
        encoder += [nn.Conv2d(self.num_hidden_features, self.num_hidden_features, kernel_size, stride, padding), nn.SiLU()]
        encoder += [nn.Conv2d(self.num_hidden_features, self.num_latent_features, kernel_size, stride, padding)]
        self.encoder = nn.Sequential(*encoder)



class VariationalAutoEncoder(nn.Module):
    def __init__(self, num_euv_channels, num_hidden_features, num_latent_features, layer_type="pixel"):
        super(VariationalAutoEncoder, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_hidden_features = num_hidden_features
        self.num_latent_features = num_latent_features
        self.layer_type = layer_type

        if self.layer_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.layer_type == "conv":
            kernel_size, stride, padding = 3, 1, 1
        self.conv1 = nn.Conv2d(self.num_euv_channels, self.num_hidden_features, kernel_size, stride, padding)
        self.conv2_mu = nn.Conv2d(self.num_hidden_features, num_latent_features, kernel_size, stride, padding)
        self.conv2_logvar = nn.Conv2d(self.num_hidden_features, num_latent_features, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(num_latent_features, self.num_hidden_features, kernel_size, stride, padding)
        self.conv4 = nn.Conv2d(self.num_hidden_features, self.num_euv_channels, kernel_size, stride, padding)
        self.act = nn.SiLU()

        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))


    def build(self):
        self.encoder = Encoder(self.num_euv_channels, self.num_hidden_features, self.num_latent_features*2, self.layer_type)
        self.decoder = Decoder(self.num_euv_channels, self.num_hidden_features, self.num_latent_features, self.layer_type)

    def encode(self, x):
        out = self.encoder(x)
        mu, logvar = torch.split(out, self.num_latent_features, dim=1)
        return mu, logvar
#        h = self.act(self.conv1(x))
#        return self.conv2_mu(h), self.conv2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent):
#        h = self.act(self.conv3(latent))
#        return self.conv4(h)
        return self.decoder(latent)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), latent, mu, logvar


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


class AutoEncoderLoss(nn.Module):
    def __init__(self):
        super(AutoEncoderLoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, x_recon, x):
        return self.criterion(x_recon, x)
    

class VariationalAutoEncoderLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(VariationalAutoEncoderLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.beta = beta

    def forward(self, x_recon, x, mu, logvar):
        recon_loss = self.criterion(x_recon, x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_div


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

    def forward(self, x_recon, x):
        return self.metric(x_recon, x)


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