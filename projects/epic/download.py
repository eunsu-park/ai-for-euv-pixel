import os
from glob import glob
import numpy as np
import asdf
from multiprocessing import Pool, freeze_support


def load(file_path):
    af = asdf.open(file_path)
    tree = af.tree
    aia_94 = tree["94"][None, ...]
    aia_131 = tree["131"][None, ...]
    aia_171 = tree["171"][None, ...]
    aia_193 = tree["193"][None, ...]
    aia_211 = tree["211"][None, ...]
    aia_335 = tree["335"][None, ...]
    data = np.concatenate([aia_94, aia_131, aia_171, aia_193, aia_211, aia_335], axis=0)
    af.close()
    return data


def normalize(data):
    data = np.clip(data, 0, None)
    data = np.log2(data + 1)
    data = data / 7. - 1.
    return data


def main(file_path):
    file_name = os.path.basename(file_path)
    data = load(file_path)
    data = normalize(data).astype(np.float32)

    tree = {"data":data}
    save_name = file_name
    save_dir = "/home/eunsu/Dataset/epic/train"
    save_path = os.path.join(save_dir, save_name)

    af = asdf.AsdfFile(tree)
    af.write_to(save_path)
    af.close()
    print(f"save: {save_path}")


if __name__ == "__main__" :

    freeze_support()

    list_data = glob("/storage/aisolo/aia_pixel_dl/resized/*/*.asdf")
    print(len(list_data))

    # file_path = list_data[0]
    # data = main(file_path)

    with Pool(16) as p:
        p.map(main, list_data)





