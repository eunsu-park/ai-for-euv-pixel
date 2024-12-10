import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    def initialize(self):
        # Environment
        self.parser.add_argument("--seed", type=int, default=2331,
                                 help="random seed")
        self.parser.add_argument("--data_root", type=str, 
                                 help="path to data")
        self.parser.add_argument("--save_root", type=str, 
                                 help="path to save model and result")
        self.parser.add_argument("--gpu_ids", type=str, default='0',
                                 help='gpu ids, ex) 0  0,1,2  0,2. -1 for CPU')

        # Dataset
        self.parser.add_argument("--nb_channels", type=int, default=6,
                                 help="number of input channel")
        self.parser.add_argument("--nb_workers", type=int, default=4,
                                 help="# of process for dataloader")
        self.parser.add_argument("--batch_size", type=int, default=5,
                                 help="batch size")

        # Model
        self.parser.add_argument("--nb_layers", type=int, default=4,
                                    help="number of layers")
        self.parser.add_argument("--nb_features", type=int, default=64,
                                    help="number of feature maps")
        self.parser.add_argument("--init_type", type=str, default="normal",
                                 help="network initialization [normal | xavier | kaiming | orthogonal]")
        self.parser.add_argument("--init_gain", type=float, default=0.02,
                                 help="scaling factor for normal, xavier and orthogonal.")

        # Loss, Metric
        self.parser.add_argument("--criterion", type=str, default="l1",
                                 help="loss function [l1 | l2 ]")
        self.parser.add_argument("--metric", type=str, default="l2",
                                 help="metric function [l1 | l2 ]")
        # Optimizer
        self.parser.add_argument("--lr", type=float, default=0.0002,
                                 help="initial learning rate for adam")
        self.parser.add_argument("--beta1", type=float, default=0.5,
                                 help="momentum term of adam")
        self.parser.add_argument("--beta2", type=float, default=0.999,
                                 help="momentum term of adam")

    def parse(self):
        self.initialize()
        return self.parser.parse_args()


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.parser.add_argument("--is_train", type=bool, default=True,
                                 help="train or validation/test")
        self.parser.add_argument("--phase", type=str, default="train",
                                 help="current phase, train")
        
        self.parser.add_argument("--nb_epochs", type=int, default=500,
                                 help="# of epochs with initial learning rate")
        self.parser.add_argument("--nb_epochs_decay", type=int, default=500,
                                 help="# of epochs with linearly decaying learning rate")
        self.parser.add_argument("--logging_freq", type=int, default=1000,
                                 help="logging frequency in iterations")
        self.parser.add_argument("--model_save_freq", type=int, default=100,
                                 help="model saving frequency in epochs")


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()

