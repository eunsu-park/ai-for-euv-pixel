import os
import datetime
import warnings
from glob import glob
import numpy as np
from astropy.io import fits
from skimage.transform import resize
from imageio import imsave
import asdf
from multiprocessing import Pool, freeze_support
import matplotlib.pyplot as plt


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def interpol(ndarray, new_shape, order=1, mode="constant", preserve_range=True):
    return resize(ndarray, new_shape, order=order, mode=mode, preserve_range=preserve_range)


def pad(data):
    pad_i = (4096-data.shape[0])//2
    pad_j = (4096-data.shape[1])//2
    data = np.pad(data, ((pad_i, pad_i), (pad_j, pad_j)), mode="edge")
    return data

def read_fits(file_path):
    hdul = fits.open(file_path)
    hdu = hdul[-1]
    header = hdu.header
    data = None
    if header["QUALITY"] != 0 :
        return data
    data = hdu.data
    data[np.isnan(data)] = 0.
    data = data / header["EXPTIME"]
    if data.shape != (4096, 4096):
        data = pad(data)
    return data


def signaltonoise_dB(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))


def signaltonoise(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def image_func(data):
    image = np.clip(data, 0, None)
    image = np.log2(image+1)
    image = image * (255./14.)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def main_func(load_dir, save_root):

    date = datetime.datetime.strptime(load_dir.split('/')[-1], "%Y%m%d%H")
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour

    save_dir = f"{save_root}/{year:04d}"
    save_name = f"aia.euv.{year:04d}-{month:02d}-{day:02d}-{hour:02d}-00-00"

    list_fits = glob(os.path.join(load_dir, "*.fits"))
    if len(list_fits) != 6 :
        return
    
    tree = {}
    images = {}
    
    for file_path in list_fits :
        hdu = fits.open(file_path)[-1]
        header = hdu.header
        data = hdu.data
        quality = header["QUALITY"]
        if quality != 0 :
            return
        data[np.isnan(data)] = 0.
        data = data / header["EXPTIME"]
        if data.shape != (4096, 4096):
            data = pad(data)
        data = interpol(data, (1024, 1024))
        image = image_func(data)
        tree[str(header["WAVELNTH"])] = data
        images[str(header["WAVELNTH"])] = image

    af = asdf.AsdfFile(tree)
    af.write_to(os.path.join(save_dir, f"{save_name}.asdf"))
    
    images = np.vstack((
        np.hstack((images["94"], images["131"], images["171"])),
        np.hstack((images["193"], images["211"], images["335"]))
    ))

    imsave(os.path.join(save_dir, f"{save_name}.png"), images)
    print(os.path.join(save_dir, f"{save_name}.asdf"))
    

if __name__ == "__main__" :
    freeze_support()
    warnings.filterwarnings('ignore')

    load_root = "/storage/aisolo/aia_pixel_dl_preped"
    save_root = "/storage/aisolo/aia_pixel_dl_resized"

    list_dir = glob(os.path.join(f"{load_root}", "*", "*"))
    nb_dir = len(list_dir)
    print(nb_dir)

#    main_func(list_dir[0], save_root)

    with Pool(16) as pool:
        pool.starmap(main_func, [(load_dir, save_root) for load_dir in list_dir])
        pool.close()

    # with 

    # list_fits = glob(os.path.join(f"{load_root}", "*", "*", "*.fits"))
    # nb_fits = len(list_fits)
    # print(nb_fits)

    # pool = Pool(16)
    # pool.map(main_func, list_fits)
    # pool.close()


    # main_func(list_fits[0])

    

    # date = datetime.datetime(year=2011, month=1, day=1, hour=0)



    # while date < datetime.datetime.now() :

    #     year = date.year
    #     month = date.month
    #     day = date.day
    #     hour = date.hour

    #     print(date)
    #     load_dir = f"{load_root}/{year:04d}/{year:04d}{month:02d}{day:02d}{hour:02d}"
    #     list_fits = glob(f"{load_dir}/*.fits")

    #     if len(list_fits) == 6 :
    #         save_dir = f"{save_root}/{year:04d}/{year:04d}{month:02d}{day:02d}{hour:02d}"
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         for file_path in list_fits :
    #             print(file_path)
    #             file_name = os.path.basename(file_path)
    #             save_name = file_name.split('.')
    #             save_name[-1] = "asdf"
    #             save_name = '.'.join(save_name)
    #             save_path = os.path.join(save_dir, save_name)
    #             data = main_func(file_path)
    #             tree = {"data":data}
    #             af = asdf.AsdfFile(tree)
    #             af.write_to(save_path)
    #             print(save_path)

    #     date = datetime.timedelta(hours=1)
    #     break



