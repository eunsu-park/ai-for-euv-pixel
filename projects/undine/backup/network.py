import torch
import torch.nn as nn



class UNDINE(nn.Module):
    def __init__(self, options):
        super(UNDINE, self).__init__()
        self.build(options)
        print(self.model)
        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def build(self, options):

        # model = []
        # model += [nn.Linear(options.nb_wave, options.nb_node), nn.ReLU()]
        # for i in range(options.nb_layer - 2):
        #     model += [nn.Linear(options.nb_node, options.nb_node), nn.ReLU()]
        # model += [nn.Linear(options.nb_node, options.nb_tbin)]
        # self.model = nn.Sequential(*model)

        model = []
        model += [nn.Conv2d(options.nb_wave, options.nb_node, 1, 1, 0), nn.ReLU()]
        for i in range(options.nb_layer - 2):
            model += [nn.Conv2d(options.nb_node, options.nb_node, 1, 1, 0), nn.ReLU()]
        model += [nn.Conv2d(options.nb_node, options.nb_tbin, 1, 1, 0)]
        self.model = nn.Sequential(*model)

        
    def forward(self, x):
        return self.model(x)


class Loss(nn.Module):
    def __init__(self, options):
        super(Loss, self).__init__()
        self.build(options)

    def build(self, options):
        if options.loss == 'l2':
            self.loss = nn.MSELoss()
        elif options.loss == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError(f'Loss function [{options.loss}] is not implemented')

    def forward(self, pred, target):
        return self.loss(pred, target)


if __name__ == '__main__' :

    from options import TrainOptions
    options = TrainOptions().parse()
    model = UNDINE(options)
