import torch
import torch.nn as nn



class UNDINE(nn.Module):
    def __init__(self, options):
        super(UNDINE, self).__init__()
        self.build(options)
        print(self.model)
        print('The number of parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def build(self, options):

        model = []
        model += [nn.Linear(options.nb_wave, options.nb_node), nn.ReLU()]
        for i in range(options.nb_layer - 1):
            model += [nn.Linear(options.nb_node, options.nb_node), nn.ReLU()]
        model += [nn.Linear(options.nb_node, options.nb_tbin)]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__" :

    from options import TrainOptions
    options = TrainOptions().parse()
    model = UNDINE(options)
