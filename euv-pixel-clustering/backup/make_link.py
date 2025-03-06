import os
import datetime



source_dir = "/home/eunsu/Dataset/epic/all"
destination_dir = "/home/eunsu/Dataset/epic/train"


if __name__ == "__main__" :

    date = datetime.datetime(2011, 1, 1, 0)
    while date < datetime.datetime(2020, 1, 1, 0) :

        year = date.year
        month = date.month
        day = date.day
        hour = date.hour

        file_name = f"aia.euv.{year:04d}-{month:02d}-{day:02d}-{hour:02d}-00-00.asdf"
        
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)

        if os.path.exists(source_path):
            command = f"ln -s {source_path} {destination_path}"
            os.system(command)
            print(f"make link: {destination_path}")

        date += datetime.timedelta(days=1)

    