import os
import datetime
import tempfile
from glob import glob
import warnings
import shutil
from functools import partial
from multiprocessing import Pool
import requests
import urllib3
import drms
import pickle
import astropy.units as u
import aiapy.psf
from sunpy.map import Map
from aiapy.calibrate import register, update_pointing, fix_observer_location
from aiapy.calibrate import fetch_spikes, respike
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table, get_pointing_table
import aiapy.psf
import astropy.units as u
import ssl
import numpy as np
from astropy.time import Time
urllib3.disable_warnings()
from imageio import imsave
ssl._create_default_https_context = ssl._create_unverified_context
from astropy.io import fits


def download_url(source, destination):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temporary = f.name
    
    if os.path.exists(destination):
        return False
    try:
        if not os.path.exists(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))
        # 스트리밍을 사용하여 파일 다운로드
        with requests.get(source, stream=True, verify=False) as response:
            response.raise_for_status()
            with open(temporary, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        # 임시 파일을 최종 목적지로 이동
        shutil.move(temporary, destination)
        os.system(f"chmod 777 {destination}")
        print(f"{source} -> {destination}")
        return True

    except requests.HTTPError as e:
        print(f"{source} : {e}")
        return False
    except requests.RequestException as e:
        print(f"요청 중 오류가 발생했습니다: {e}")
        return False


def get_client(email):
    client = drms.Client(email=email)
    return client
    

def request(client, dt_start):
    dt_end = dt_start + datetime.timedelta(minutes=1)    
    str_start = dt_start.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    str_end = dt_end.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    query_str = f"aia.lev1_euv_12s[{str_start}-{str_end}]"
    print(query_str)
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()
    return export_request


def download(export_request, out_dir):
    done = False
    source_and_destination = []

    for url in export_request.urls.url:
        filename = os.path.basename(url)
        destination = os.path.join(out_dir, filename)
        source_and_destination.append((url, destination))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.system(f"chmod 777 {out_dir}")

    while done is False:
        with Pool(8) as p:
            p.starmap(download_url, source_and_destination)
        not_downloaded = []
        for source, destination in source_and_destination:
            if not os.path.exists(destination):
                not_downloaded.append((source, destination))
        if len(not_downloaded) == 0:
            done = True
        else:
            source_and_destination = not_downloaded


def save_correction_table(file_path):
    correction_table = get_correction_table()
    with open(file_path, 'wb') as f:
        pickle.dump(correction_table, f)


def load_correction_table(file_path):
    if not os.path.exists(file_path):
        save_correction_table(file_path)
    with open(file_path, 'rb') as f :
        return pickle.load(f)

def save_pointing_table(file_path):
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime.now()
    pointing_table = get_pointing_table(start, end)
    with open(file_path, 'wb') as f:
        pickle.dump(pointing_table, f)


def load_pointing_table(file_path):
    if not os.path.exists(file_path):
        save_pointing_table(file_path)
    with open(file_path, 'rb') as f :
        return pickle.load(f)


def save_psfs(file_path):
    waves = [94, 131, 171, 193, 211, 304, 335]
    psfs = {}
    for wave in waves:
        psf = aiapy.psf.psf(wave*u.angstrom)
        psfs[str(int(wave))] = psf
    with open(file_path, 'wb') as f:
        pickle.dump(psfs, f)


def load_psfs(file_path):
    if not os.path.exists(file_path):
        save_psfs(file_path)
    with open(file_path, 'rb') as f :
        return pickle.load(f)


def preparation(file_path,
    do_update_pointing=False, pointing_table=None,
    do_fix_observer_location=False,
    do_respike=False,
    do_deconvolve=False, psfs=None,
    do_correct_degradation=False, correction_table=None) :

    ## Load AIA map
    aia_map = Map(file_path)

    ## Update pointing
    if do_update_pointing :
        if pointing_table is None :
            aia_map = update_pointing(aia_map)
        else :
            aia_map = update_pointing(aia_map, pointing_table=pointing_table)

    ## Fix observer location
    if do_fix_observer_location :
        aia_map = fix_observer_location(aia_map)

    ## Respike
    if do_respike :
        positions, values = fetch_spikes(aia_map)
        aia_map = respike(aia_map, spikes=(positions, values))

    ## Deconvolve
    if do_deconvolve :
        wavelnth = aia_map.meta["wavelnth"]
        if wavelnth in [94, 131, 171, 193, 211, 304, 335] :
            if psfs is None :
                psf = aiapy.psf.psf(aia_map.wavelength)
            else :
                psf = psfs[str(int(wavelnth))]
            aia_map = aiapy.psf.deconvolve(aia_map, psf=psf)        

    ## Register
    aia_map = register(aia_map)

    ## Correct degradation
    if do_correct_degradation :
        if wavelnth in [94, 131, 171, 193, 211, 304, 335] :
            if correction_table is None :
                aia_map = correct_degradation(aia_map)
            else :
                aia_map = correct_degradation(aia_map, correction_table=correction_table)

    # data = aia_map.data
    # meta = aia_map.meta
    # if data.shape != (4096, 4096) :
    #     i_pad = (4096 - data.shape[0]) // 2
    #     j_pad = (4096 - data.shape[1]) // 2
    #     data = np.pad(data, ((i_pad, i_pad), (j_pad, j_pad)), mode="constant", constant_values=0)
    #     aia_map = Map(data, meta)

    return aia_map

def get_preparation_func(do_update_pointing, pointing_table,
    do_fix_observer_location,
    do_respike,
    do_deconvolve, psfs,
    do_correct_degradation, correction_table) :

    return partial(preparation,
        do_update_pointing=do_update_pointing, pointing_table=pointing_table,
        do_fix_observer_location=do_fix_observer_location,
        do_respike=do_respike,
        do_deconvolve=do_deconvolve, psfs=psfs,
        do_correct_degradation=do_correct_degradation, correction_table=correction_table)


def save_image(aia_map, save_path):
    data = aia_map.data
    data = np.clip(data, 0, None)
    data = np.log2(data+1)
    data = data * (255./14.)
    image = np.clip(data, 0, 255).astype(np.uint8)
    imsave(save_path, image)


def select_only_one(dt_start, data_root):

    list_fits_final = []
    waves = (94, 131, 171, 193, 211, 335)

    year = dt_start.year
    month = dt_start.month
    day = dt_start.day
    hour = dt_start.hour
    minute = dt_start.minute
    second = dt_start.second

    for wave in waves :
        dir_path = os.path.join(data_root, f"{wave:d}", f"{year:04d}", f"{year:04d}{month:02d}{day:02d}")
        file_pattern = f"aia.{year:04d}-{month:02d}-{day:02d}-{hour:02d}-*-*.{wave:d}.fits"
        list_fits = sorted(glob(os.path.join(dir_path, file_pattern)))
        nb_fits = len(list_fits)
        if nb_fits > 0 :
            file_path = list_fits[0]
            file_name = os.path.basename(file_path)
            _, date, _, _ = file_name.split(".")
            f_year, f_month, f_day, f_hour, f_minute, f_second = date.split("-")
            datetime_fits = datetime.datetime(int(f_year), int(f_month), int(f_day), int(f_hour), int(f_minute), int(f_second))
            # hdu = fits.open(file_path)[-1]
            # header = hdu.header
            # t_rec = header["T_REC"]
            # datetime_fits = Time(t_rec).datetime
            datetime_dif = abs(datetime_fits - dt_start)
            if datetime_dif.total_seconds() < 12 :
                list_fits_final.append(file_path)

    if len(list_fits_final) == len(waves) :
        return list_fits_final
    else :
        return []
