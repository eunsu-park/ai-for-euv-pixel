import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_euv_channels, num_latent_features, model_type="pixel"):
        super(Encoder, self).__init__()
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
        model += [nn.Conv2d(self.num_euv_channels, 1024, kernel_size, stride, padding), nn.SiLU()]
        model += [nn.Conv2d(1024, self.num_latent_features, kernel_size, stride, padding), nn.Sigmoid()]
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
