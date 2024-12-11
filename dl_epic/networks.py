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
        model += [nn.Conv2d(in_channels, self.nb_features, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)] # Sigmoid?
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


class EncoderDecoer(nn.Module):
    def __init__(self, nb_channels, nb_features, nb_layers):
        super(EncoderDecoer, self).__init__()
        self.encoder = Encoder(nb_channels, nb_features, nb_layers)
        self.decoder = Decoder(nb_features, nb_channels, nb_layers)

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out


def define_network(options):
    network = EncoderDecoer(
        options.nb_channels, options.nb_features, options.nb_layers)
    return network


def define_criterion(options):
    if options.criterion == "l1" :
        criterion = nn.L1Loss()
    elif options.criterion == "l2" :
        criterion = nn.MSELoss()
    else :
        raise NotImplementedError("Loss function [%s] is not implemented" % options.criterion)
    return criterion


def define_optimizer(options, network):
    optimizer = torch.optim.Adam(
        network.parameters(), lr=options.lr,
        betas=(options.beta1, options.beta2))
    return optimizer


def define_scheduler(options, optimizer):
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + 1 - options.nb_epochs) / float(options.nb_epochs_decay + 1)    
    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_rule)


if __name__ == "__main__" :

    from options import TrainOptions
    options = TrainOptions().parse()

    network = define_network(options)
    print(network)
    inp = torch.randn(1, options.nb_channels, 1024, 1024)
    feat = network.encoder(inp)
    out = network(inp)
    print(inp.size(), feat.size(), out.size())

    criterion = define_criterion(options)
    print(criterion)

    optimizer = define_optimizer(options, network)
    print(optimizer)

    scheduler = define_scheduler(options, optimizer)
    print(scheduler)



