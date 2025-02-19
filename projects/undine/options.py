import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--seed', type=int, default=0,
                                 help='Random seed')
        
        self.parser.add_argument('--data_file_path', type=str,
                                 help='Path to data file')
        self.parser.add_argument('--save_root', type=str,
                                 help='Root directory to save results')
               
        self.parser.add_argument('--nb_wave', type=int, default=6,
                                 help='Number of wavelengths')
        self.parser.add_argument('--t_min', type=float, default=0.1,
                                 help='Minimum DEM temperature in LogT')
        self.parser.add_argument('--t_max', type=float, default=1.0,
                                 help='Maximum DEM temperature in LogT')
        self.parser.add_argument('--nb_tbin', type=int, default=43,
                                 help='Number of temperature bins')

        self.parser.add_argument('--nb_layer', type=int, default=5,
                                 help='Number of hidden layers')  
        self.parser.add_argument('--nb_node', type=int, default=16384,
                                 help='Number of nodes in hidden layers')

    def parse(self):
        return self.parser.parse_args()


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=True,
                                 help='Flag for training')
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='Checkpoint to resume')

        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='batch size')
        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help='# of process for dataloader')
        
        self.parser.add_argument('--loss', type=str, default='l2',
                                 help='Loss function')
        self.parser.add_argument('--optimizer', type=str, default='adam',
                                 help='Optimizer')

        self.parser.add_argument('--lr', type=float, default=0.0002,
                                 help='initial learning rate')
        self.parser.add_argument('--beta1', type=float, default=0.5,
                                 help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999,
                                 help='momentum term of adam')
        self.parser.add_argument('--eps', type=float, default=1e-8,
                                 help='momentum term of adam')
        self.parser.add_argument('--weight_decay', type=float, default=0.001,
                                 help='weight decay')
        self.parser.add_argument('--nb_epochs', type=int, default=50,
                                 help='# of epochs with initial learning rate')
        self.parser.add_argument('--nb_epochs_decay', type=int, default=50,
                                 help='# of epochs with linearly decaying learning rate')

        self.parser.add_argument('--report_freq', type=int, default=1000,
                                 help='report frequency in iterations')
        self.parser.add_argument('--save_freq', type=int, default=10,
                                 help='save frequency in epochs')


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()