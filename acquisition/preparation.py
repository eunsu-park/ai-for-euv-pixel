import os
import datetime
import argparse
from utils import load_correction_table, load_pointing_table, load_psfs
from utils import get_preparation_func, save_image, select_only_one



def main(args):

    correction_table_file_path = f"{args.load_root}/correction_table.pkl"
    correction_table = load_correction_table(correction_table_file_path)

    pointing_table_file_path = f"{args.load_root}/pointing_table.pkl"
    pointing_table = load_pointing_table(pointing_table_file_path)

    psfs_file_path = f"{args.load_root}/psfs.pkl"
    psfs = load_psfs(psfs_file_path)

    preparation_func = get_preparation_func(
        do_update_pointing=True, pointing_table=pointing_table,
        do_fix_observer_location=True,
        do_respike=False,
        do_deconvolve=True, psfs=psfs,
        do_correct_degradation=True, correction_table=correction_table)
   
    load_dir = f"{args.load_root}/{args.year:04d}/{args.year:04d}{args.month:02d}{args.day:02d}{args.hour:02d}"
    save_dir = f"{args.save_root}/{args.year:04d}/{args.year:04d}{args.month:02d}{args.day:02d}{args.hour:02d}"

    dt_start = datetime.datetime(year=args.year, month=args.month,
                                 day=args.day, hour=args.hour)

    list_data = select_only_one(dt_start, load_dir)

    if list_data is not None :
        preparation_flag = f"{save_dir}/done.preparation"
        if not os.path.exists(preparation_flag):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            for file_path in list_data :
                file_name = os.path.basename(file_path)
                save_path = f"{save_dir}/{file_name}"
                if not os.path.exists(save_path) :
                    result = preparation_func(file_path=file_path)
                    result.save(save_path, overwrite=True)
                    save_image(result, f"{save_path}.png")
                else:
                    pass
            with open(preparation_flag, "w") as f:
                f.write("done")
        else :
            pass


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_root", type=str, help="Root directory of original files")
    parser.add_argument("--save_root", type=str, help="Root directory of preped files")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--day", type=int)
    parser.add_argument("--hour", type=int)
    args = parser.parse_args()

    main(args)