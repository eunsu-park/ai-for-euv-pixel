import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, nb_channels, nb_features, nb_layers):
        super(Encoder, self).__init__()
        self.nb_channels = nb_channels
        self.nb_features = nb_features
        self.nb_layers = nb_layers
        self.build()

    def build(self):
        model = []
        in_channels = self.nb_channels
        out_channels = 64
        for i in range(self.nb_layers - 1):
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)]
            in_channels = out_channels
            out_channels = 2*out_channels
        model += [nn.Conv2d(in_channels, self.nb_features, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, nb_features, nb_channels, nb_layers):
        super(Decoder, self).__init__()
        self.nb_features = nb_features
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.build()

    def build(self):
        model = []
        in_channels = self.nb_features
        out_channels = 64 * (2**(self.nb_layers - 2))
        for i in range(self.nb_layers - 1):
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)]
            in_channels = out_channels
            out_channels = out_channels//2
        model += [nn.Conv2d(in_channels, self.nb_channels, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__" :

    from options import TrainOptions
    options = TrainOptions().parse()

    encoder = Encoder(options.nb_channels, options.nb_features, options.nb_layers)
    print(encoder)
    decoder = Decoder(options.nb_features, options.nb_channels, options.nb_layers)
    print(decoder)
    inp = torch.randn(1, options.nb_channels, 1024, 1024)
    feat = encoder(inp)
    out = decoder(feat)
    print(inp.size(), feat.size(), out.size())



