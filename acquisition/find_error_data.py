import os
import shutil
import datetime
from glob import glob
from sunpy.map import Map
from astropy.io import fits
from multiprocessing import Pool, freeze_support


def main_func(file_path):
    file_name = os.path.basename(file_path)
    error_data_dir = "/storage/obsdata/sdo/aia/error_data"

    temporary = f"/home/eunsu/tmp/{file_name}"
    shutil.copy(file_path, temporary)

    try :
        M = Map(temporary)
        meta = M.meta
        data = M.data
        if data.shape != (4096, 4096) :
            shutil.move(file_path, f"{error_data_dir}/{file_name}")
            print("sunpy", file_path, data.shape)
            return

    except Exception as e:
        print("sunpy", file_path, e)
        shutil.move(file_path, f"{error_data_dir}/{file_name}")
        return 

    try :
        hdul = fits.open(temporary)
        header = hdul[-1].header
        data = hdul[-1].data
        if data.shape != (4096, 4096) :
            print("astropy", file_path, data.shape)
            shutil.move(file_path, f"{error_data_dir}/{file_name}")
            return 
    except Exception as e:
        print("astropy", file_path, e)
        shutil.move(file_path, f"{error_data_dir}/{file_name}")
        return 
    
    os.remove(temporary)


if __name__ == '__main__':

    waves = (94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500)
    year, month, day = 2010, 7, 19
    date = datetime.datetime(year, month, day)

    while date < datetime.datetime.now():
        year = date.year
        month = date.month
        day = date.day
        list_data = []
        for wave in waves :
            dir_path = f"/storage/obsdata/sdo/aia/{wave}/{year:04d}/{year:04d}{month:02d}{day:02d}"
            list_data += glob(f"{dir_path}/*.fits")
        print(date, len(list_data))
        with Pool(64) as p:
            p.map(main_func, list_data)

        date += datetime.timedelta(days=1)
