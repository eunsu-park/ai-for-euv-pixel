import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--seed", type=int, default=0,help="Random seed")
        
        self.parser.add_argument(
            "--data_root", type=str, help="Root directory of data")
        self.parser.add_argument(
            "--save_root", type=str, help="Root directory to save results")
               
        self.parser.add_argument(
            "--nb_wave", type=int, default=6, help="Number of wavelengths")
        self.parser.add_argument(
            "--t_min", type=float, default=0.1, help="Minimum DEM temperature")
        self.parser.add_argument(
            "--t_max", type=float, default=1.0, help="Maximum DEM temperature")
        self.parser.add_argument(
            "--nb_tbin", type=int, default=43, help="Number of temperature bins")

        self.parser.add_argument(
            "--nb_layer", type=int, default=3, help="Number of hidden layers")  
        self.parser.add_argument(
            "--nb_node", type=int, default=16384, help="Number of nodes in hidden layers")

        

        
        


        self.initialized = True
        
    def parse(self):
        if not self.initialized:
            self.initialize()
        return self.parser.parse_args()


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()