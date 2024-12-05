import os
from glob import glob
import datetime
import argparse
from astropy.io import fits
from astropy.time import Time
from utils import load_correction_table, load_pointing_table, load_psfs
from utils import get_preparation_func, save_image, select_only_one


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    correction_table_file_path = f"{args.table_path}/correction_table.pkl"
    correction_table = load_correction_table(correction_table_file_path)

    pointing_table_file_path = f"{args.table_path}/pointing_table.pkl"
    pointing_table = load_pointing_table(pointing_table_file_path)

    psfs_file_path = f"{args.table_path}/psfs.pkl"
    psfs = load_psfs(psfs_file_path)

    preparation_func = get_preparation_func(
        do_update_pointing=True, pointing_table=pointing_table,
        do_fix_observer_location=True,
        do_respike=False,
        do_deconvolve=True, psfs=psfs,
        do_correct_degradation=True, correction_table=correction_table)

    dt_start = datetime.datetime(args.year, 1, 1, 0)
    dt_end = datetime.datetime(args.year+1, 1, 1, 0)

    while dt_start < dt_end :

        save_dir = f"{args.save_root}/{dt_start.year:04d}/{dt_start.year:04d}{dt_start.month:02d}{dt_start.day:02d}{dt_start.hour:02d}"

        prep_flag = f"{save_dir}/done.prep"
        if not os.path.exists(prep_flag) :
            list_fits = select_only_one(dt_start, args.load_root)
            nb_fits = len(list_fits)
            print(dt_start, nb_fits)

            if nb_fits == 6 :
                try :
                    m_preped = []
                    list_save_path = []
                    for n in range(6):
                        file_path = list_fits[n]
                        save_path = f"{save_dir}/{os.path.basename(file_path)}"
                        list_save_path.append(save_path)
                        result = preparation_func(file_path=file_path)
                        m_preped.append(result)
                    if not os.path.exists(save_dir) :
                        os.makedirs(save_dir, exist_ok=True)
                    for n in range(6):
                        m_preped[n].save(list_save_path[n], overwrite=False)
                        # save_image(m_preped[n], f"{list_save_path[n]}.png")
                    os.system(f"touch {prep_flag}")
                except :
                    pass
        dt_start += datetime.timedelta(hours=1)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--table_path", type=str, default="/home/eunsu/aia_tables", help="Path to the table")
    parser.add_argument("--load_root", type=str, default="/storage/obsdata/sdo/aia", help="Root directory of original files")
    parser.add_argument("--save_root", type=str, default="/storage/aisolo/aia_pixel_dl_preped", help="Root directory of preped files")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--day", type=int)
    parser.add_argument("--hour", type=int)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    main(args)