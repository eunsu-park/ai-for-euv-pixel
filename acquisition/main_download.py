import ssl
from utils import request, download


def manual(args):
    pass

def auto(args):
    pass

def scan(args) :
    pass


if __name__ == "__main__" :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="manual", help="Mode, scan, auto or manual")
    parser.add_argument("--email", type=str, default="eunsupark@kasi.re.kr")
    parser.add_argument("--data_root", type=str, default="/path/to/data", help="Data save root")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--day", type=int)
    parser.add_argument("--hour", type=int)
    args = parser.parse_args()
