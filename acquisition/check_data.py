import os
from glob import glob
from astropy.io import fits


data_root = "/Volumes/870EVO_1"
year, month, day, hour = 2011, 1, 1, 0
data_dir = f"{data_root}/{year:04d}/{year:04d}{month:02d}{day:02d}{hour:02d}"

list_fits = sorted(glob(f"{data_dir}/*lev1.fits"))
nb_fits = len(list_fits)
print(nb_fits)

for file_path in list_fits :
    file_name = os.path.basename(file_path)
    date = file_name.split('.')[2]
    hdu = fits.open(file_path)[-1]
    header = hdu.header
    t_rec = header["T_REC"]
    print(date, t_rec)


#aia.lev1_euv_12s.2011-01-01T015926Z.211.image_lev1.fits