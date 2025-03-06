import asdf
from glob import glob
import numpy as np
from multiprocessing import Pool, freeze_support

def read_data(file_path):
    af = asdf.open(file_path)
    tree = af.tree
    data = tree["data"]
    data = np.transpose(data, (1, 2, 0))
    data = np.expand_dims(data, axis=0)
    return data

if __name__ == "__main__" :

    list_data = glob("/home/eunsu/Dataset/epic/test/*.asdf")
    nb_data = len(list_data)
    print(f"nb_data: {nb_data}")

    with Pool(16) as p:
        data = p.map(read_data, list_data)

    data = np.concatenate(data, axis=0)
    print(data.shape)

    np.save("/home/eunsu/Dataset/epic/test.npy", data)
