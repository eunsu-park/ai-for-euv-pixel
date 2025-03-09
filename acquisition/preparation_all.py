import os
from glob import glob
import datetime
import argparse
import warnings
from astropy.io import fits
from astropy.time import Time
from utils import load_correction_table, load_pointing_table, load_psfs
from utils import get_preparation_func, save_image, select_only_one

warnings.filterwarnings("ignore")

WAVES = (94, 131, 171, 193, 211, 304, 335)
NUM_WAVES = len(WAVES)
START_YEAR = 2011
START_MONTH = 1
START_DAY = 1
START_HOUR = 0


def core(dt_start, load_root, save_root, preparation_func) :

    print(dt_start)

    year = dt_start.year
    month = dt_start.month
    day = dt_start.day
    hour = dt_start.hour

    load_dir = f"{load_root}/{year:04d}/{year:04d}{month:02d}{day:02d}{hour:02d}"
    save_dir = f"{save_root}/{year:04d}/{year:04d}{month:02d}{day:02d}{hour:02d}"

    download_flag = f"{load_dir}/done.download"
    if not os.path.exists(download_flag) :
        print(f"{dt_start}: Download is not done")
        return

    prep_flag = f"{save_dir}/done.prep"
    if os.path.exists(prep_flag) :
        print(f"{dt_start}:Preparation is already done")
        return
    
    list_fits = select_only_one(dt_start, args.load_root, WAVES)
    if len(list_fits) < NUM_WAVES :
        print(f"{dt_start}: Not enough files")
        return

    m_preped = []
    list_save_path = []

    for n in range(NUM_WAVES):
        try :
            file_path = list_fits[n]
            save_path = f"{save_dir}/{os.path.basename(file_path)}"
            result = preparation_func(file_path=file_path)
            list_save_path.append(save_path)
            m_preped.append(result)
        except :
            pass

    if len(m_preped) < NUM_WAVES :
        print(f"{dt_start}: Not enough preped files")
        return
    
    if len(list_save_path) < NUM_WAVES :
        print(f"{dt_start}: Not enough save paths")
        return
    
    if not os.path.exists(save_dir) :
        os.makedirs(save_dir, exist_ok=True)


    for n in range(NUM_WAVES):
        m_preped[n].save(list_save_path[n], overwrite=True)
        save_image(m_preped[n], f"{list_save_path[n]}.png")
    os.system(f"touch {prep_flag}")


    # try :
    #     m_preped = []
    #     list_save_path = []
    #     for n in range(NUM_WAVES):
    #         file_path = list_fits[n]
    #         save_path = f"{save_dir}/{os.path.basename(file_path)}"
    #         list_save_path.append(save_path)
    #         result = preparation_func(file_path=file_path)
    #         m_preped.append(result)

    #     if not os.path.exists(save_dir) :
    #         os.makedirs(save_dir, exist_ok=True)

    #     for n in range(NUM_WAVES):
    #         m_preped[n].save(list_save_path[n], overwrite=True)
    #         save_image(m_preped[n], f"{list_save_path[n]}.png")
    #     os.system(f"touch {prep_flag}")
    # except :
    #     pass


def run(args):

    if (args.year is None) or (args.month is None) or (args.day is None) or (args.hour is None) :
        args.year = START_YEAR
        args.month = START_MONTH
        args.day = START_DAY
        args.hour = START_HOUR

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    correction_table_file_path = f"{args.table_path}/correction_table.pkl"
    correction_table = load_correction_table(correction_table_file_path)

    pointing_table_file_path = f"{args.table_path}/pointing_table.pkl"
    pointing_table = load_pointing_table(pointing_table_file_path)

    psfs_file_path = f"{args.table_path}/psfs.pkl"
    psfs = load_psfs(psfs_file_path)

    preparation_func = get_preparation_func(
        do_update_pointing=True, pointing_table=pointing_table,
        do_respike=False,
        do_deconvolve=True, psfs=psfs,
        do_correct_degradation=True, correction_table=correction_table)

    dt_start = datetime.datetime(args.year, 1, 1, 0)
    now = datetime.datetime.now(datetime.timezone.utc)
    dt_end = datetime.datetime(now.year, now.month, now.day, now.hour)

    n = 0
    while dt_start < dt_end :
        core(dt_start, args.load_root, args.save_root, preparation_func)
        dt_start += datetime.timedelta(days=1)
        n += 1
        # if n == 10 :
        #     break



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--table_path", type=str, default="/home/eunsu/Data/pixel/utils", help="Path to the table")
    parser.add_argument("--load_root", type=str, default="/home/eunsu/Data/pixel/original", help="Root directory of original files")
    parser.add_argument("--save_root", type=str, default="/home/eunsu/Data/pixel/preped", help="Root directory of preped files")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--day", type=int)
    parser.add_argument("--hour", type=int)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    run(args)