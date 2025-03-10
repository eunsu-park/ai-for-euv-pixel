import argparse


class Options:
    def __init__(self) :
        self.parser = argparse.ArgumentParser()

        # Options for all phases
        self.parser.add_argument("--seed", type=int,
                                 default=250104, help="random seed")
        self.parser.add_argument("--device", type=str,
                                 default="cuda", help="device")
        self.parser.add_argument("--phase", type=str,
                                 choices=["train", "test", "clustering"],
                                 help="phase")
        self.parser.add_argument("--data_file_path", type=str,
                                 help="Path to data file")
        self.parser.add_argument("--response_file_path", type=str,
                                 default="./data/undine_params.h5", help="Path to response file")
        self.parser.add_argument("--save_root", type=str,
                                 help="save directory")
        self.parser.add_argument("--experiment_name", type=str,
                                 help="experiment name")        
        self.parser.add_argument("--model_type", type=str,
                                 choices=["pixel", "convolution"],
                                 default="pixel", help="model type")
        self.parser.add_argument("--waves", type=int, nargs="+",
                                 default=[94, 131, 171, 193, 211, 335],
                                 help="wavelengths")
        self.parser.add_argument("--num_temperature_bins", type=int,
                                 default=43, help="number of temperature bins")
        self.parser.add_argument("--min_temperature", type=float,
                                 default=5, help="minimum temperature in Log T")
        self.parser.add_argument("--max_temperature", type=float,
                                 default=6, help="maximum temperature in Log T")
        self.parser.add_argument("--init_type", type=str,
                                 choices=["normal", "xavier", "kaiming", "orthogonal"],
                                 default="normal", help="initialization type")
        self.parser.add_argument("--model_path", type=str,
                                 default="", help="model path")

        # Options for training
        self.parser.add_argument("--loss_function", type=str,
                                 choices=["mse", "mae"],
                                 default="mse", help="loss function")
        self.parser.add_argument("--metric_function", type=str,
                                 choices=["mse", "mae"],
                                 default="mae", help="metric function")
        self.parser.add_argument("--lr", type=float,
                                 default=0.0002, help="learning rate")
        self.parser.add_argument("--beta1", type=float,
                                 default=0.5, help="beta1 parameter of Adam optimizer")
        self.parser.add_argument("--beta2", type=float,
                                 default=0.999, help="beta2 parameter of Adam optimizer")
        self.parser.add_argument("--target_threshold", type=float,
                                 default=0.1, help="target threshold")
        self.parser.add_argument("--convegence_threshold", type=float,
                                 default=0.001, help="convergence threshold")
        self.parser.add_argument("--max_iteration", type=int,
                                 default=1000, help="number of iterations")
        self.parser.add_argument("--snapshot_interval", type=int,
                                 default=100, help="snapshot interval")


    def parse(self):
        args = self.parser.parse_args()
        args.num_euv_channels = len(args.waves)
        # self.parser.add_argument("--num_euv_channels", type=int,
        #                          default=6, help="number of EUV channels")
        # return self.parser.parse_args()
        return args
