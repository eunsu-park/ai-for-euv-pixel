import os
import argparse
import datetime
from utils import get_client, request, download


def main(args):
    client = get_client(args.email)
    dt_start = datetime.datetime(args.year, 1, 1, 0)
    dt_end = datetime.datetime(args.year+1, 1, 1, 0)

    while dt_start < dt_end :
        print(dt_start)
        year = dt_start.year
        month = dt_start.month
        day = dt_start.day
        hour = dt_start.hour
        out_dir = f"{args.save_root}/{year:04d}/{year:04d}{month:02d}{day:02d}{hour:02d}"
        download_flag = f"{out_dir}/done.download"
        if not os.path.exists(download_flag):
            export_request = request(client, dt_start)
            download(export_request, out_dir=out_dir)
            os.system(f"touch {download_flag}")
        else :
            pass
        dt_start += datetime.timedelta(hours=1)
    
     

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str, default="eunsupark@kasi.re.kr")
    parser.add_argument("--save_root", type=str, help="Output directory")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--day", type=int)
    parser.add_argument("--hour", type=int)
    args = parser.parse_args()
    main(args)


#/Volumes/870EVO_1