import argparse


class BaseOptions:
    def __init__(self) :
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--seed", type=int,
                                 default=250104, help="random seed")
        self.parser.add_argument("--device", type=str,
                                 default="cuda", help="device")
        self.parser.add_argument("--data_root", type=str,
                                 help="data root directory")
        self.parser.add_argument("--save_root", type=str,
                                 help="save directory")
        self.parser.add_argument("--batch_size", type=int,
                                 default=1, help="batch size")
        self.parser.add_argument("--num_workers", type=int,
                                 default=8, help="number of workers")
        self.parser.add_argument("--model_type", type=str,
                                 choices=["pixel", "convolution"],
                                 default="pixel", help="model type")
        self.parser.add_argument("--num_euv_channels", type=int,
                                 default=6, help="number of EUV channels")
        self.parser.add_argument("--num_latent_features", type=int,
                                 default=50, help="number of latent features")
        self.parser.add_argument("--init_type", type=str,
                                 choices=["normal", "xavier", "kaiming", "orthogonal"],
                                 default="normal", help="initialization type")

    def parse(self):
        return self.parser.parse_args()


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.parser.add_argument("--is_train", type=bool,
                                 default=True, help="train or test")
        self.parser.add_argument("--model_path", type=str,
                                 default="", help="model path")
        self.parser.add_argument("--lr", type=float,
                                 default=0.0002, help="learning rate")
        self.parser.add_argument("--beta1", type=float,
                                 default=0.5, help="beta1 parameter of Adam optimizer")
        self.parser.add_argument("--beta2", type=float,
                                 default=0.999, help="beta2 parameter of Adam optimizer")
        self.parser.add_argument("--n_epochs", type=int,
                                 default=10, help="number of epochs")
        self.parser.add_argument("--report_freq", type=int,
                                 default=1000, help="report frequency in iterations")
        self.parser.add_argument("--save_freq", type=int,
                                 default=1, help="save frequency in epochs")


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument("--is_train", type=bool,
                                 default=False, help="train or test")
        self.parser.add_argument("--model_path", type=str,
                                 default="", help="model path")


class ClusteringOptions(BaseOptions):
    def __init__(self):
        super(ClusteringOptions, self).__init__()
        self.parser.add_argument("--n_clusters", type=int,
                                 default=10, help="number of clusters")
        self.parser.add_argument("--n_init", type=int,
                                 default=10, help="number of initializations")
        self.parser.add_argument("--max_iter", type=int,
                                 default=300, help="maximum number of iterations")
        self.parser.add_argument("--tol", type=float,
                                 default=1e-4, help="tolerance")
        self.parser.add_argument("--report_freq", type=int,
                                 default=1000, help="report frequency in iterations")
        self.parser.add_argument("--save_freq", type=int,
                                 default=1, help="save frequency in epochs")