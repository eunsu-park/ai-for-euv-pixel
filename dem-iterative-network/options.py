import argparse


class BaseOptions:
    def __init__(self) :
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--seed", type=int,
                                 default=250104, help="random seed")
        self.parser.add_argument("--device", type=str,
                                 default="cuda", help="device")
        self.parser.add_argument("--data_file_path", type=str,
                                 help="Path to data file")
        self.parser.add_argument("--response_file_path", type=str,
                                 default="./data/undine_params.h5", help="Path to response file")
        self.parser.add_argument("--save_root", type=str,
                                 help="save directory")
        self.parser.add_argument("--batch_size", type=int,
                                 default=4, help="batch size")
        self.parser.add_argument("--num_workers", type=int,
                                 default=8, help="number of workers")
        self.parser.add_argument("--model_type", type=str,
                                 choices=["pixel", "convolution"],
                                 default="pixel", help="model type")
        self.parser.add_argument("--num_euv_channels", type=int,
                                 default=6, help="number of EUV channels")
        self.parser.add_argument("--num_temperature_bins", type=int,
                                 default=43, help="number of temperature bins")
        self.parser.add_argument("--min_temperature", type=float,
                                 default=5, help="minimum temperature in Log T")
        self.parser.add_argument("--max_temperature", type=float,
                                 default=6, help="maximum temperature in Log T")
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
        self.parser.add_argument("--n_iterations", type=int,
                                 default=1000, help="number of iterations")
        self.parser.add_argument("--report_freq", type=int,
                                 default=100, help="report frequency in iterations")
        self.parser.add_argument("--save_freq", type=int,
                                 default=100, help="save frequency in iterations")


class TestOptions(BaseOptions):
    def __init__(self):

        super(TestOptions, self).__init__()
        self.parser.add_argument("--is_train", type=bool,
                                 default=False, help="train or test")
        self.parser.add_argument("--model_path", type=str,
                                 default="", help="model path")
