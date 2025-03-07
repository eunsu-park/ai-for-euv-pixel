import time
import numpy as np
from options import Options
from main import EPIC


def test():
    options = Options().parse()
    options.phase = "test"

    model = EPIC(options)

    if not options.model_path :
        raise ValueError("The model path must be provided for testing.")

    model.load_networks(options.model_path)
    model.test()


if __name__ == "__main__" :
    test()