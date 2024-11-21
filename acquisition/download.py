import datetime
from utils import get_client, request, download


def main(args):
    client = get_client(args.email)
    dt_start = datetime.datetime(args.year, args.month, args.day, args.hour)
    export_request = request(client, dt_start)
    download(export_request, out_dir=args.out_dir)
     

if __name__ == "__main__" :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str, default="eunsupark@kasi.re.kr")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--day", type=int)
    parser.add_argument("--hour", type=int)
    args = parser.parse_args()
    main(args)
