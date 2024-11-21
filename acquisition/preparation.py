import os
import pickle
import warnings
import datetime
from glob import glob
from multiprocessing import Pool, freeze_support
from functools import partial

import ssl
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import aiapy.psf
from sunpy.map import Map
from aiapy.calibrate import register, update_pointing, fix_observer_location
from aiapy.calibrate import fetch_spikes, respike
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table, get_pointing_table


def save_psfs(file_path):
    waves = [94, 131, 171, 193, 211, 304, 335]
    psfs = {}
    for wave in waves:
        psf = aiapy.psf.psf(wave*u.angstrom)
        psfs[str(int(wave))] = psf
    with open(file_path, 'wb') as f:
        pickle.dump(psfs, f)


def save_correction_table(file_path):
    correction_table = get_correction_table()
    with open(file_path, 'wb') as f:
        pickle.dump(correction_table, f)


def save_pointing_table(file_path):
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime.now()
    pointing_table = get_pointing_table(start, end)
    with open(file_path, 'wb') as f:
        pickle.dump(pointing_table, f)


def aia_prep(file_path,
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

    return aia_map


def get_aia_prep_func(do_update_pointing, pointing_table,
    do_fix_observer_location,
    do_respike,
    do_deconvolve, psfs,
    do_correct_degradation, correction_table) :

    return partial(aia_prep,
        do_update_pointing=do_update_pointing, pointing_table=pointing_table,
        do_fix_observer_location=do_fix_observer_location,
        do_respike=do_respike,
        do_deconvolve=do_deconvolve, psfs=psfs,
        do_correct_degradation=do_correct_degradation, correction_table=correction_table)


def main(file_path, save_path,
    do_update_pointing, pointing_table,
    do_fix_observer_location,
    do_respike,
    do_deconvolve, psfs,
    do_correct_degradation, correction_table) :

    aia_map = aia_prep(file_path,
        do_update_pointing=do_update_pointing, pointing_table=pointing_table,
        do_fix_observer_location=do_fix_observer_location,
        do_respike=do_respike,
        do_deconvolve=do_deconvolve, psfs=psfs,
        do_correct_degradation=do_correct_degradation, correction_table=correction_table)
    
    aia_map.save(save_path)
    print(f"Saved {save_path}")


if __name__ == "__main__" :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_update_pointing", type=bool, default=True)
    parser.add_argument("--pointing_table", type=str, default="/Users/eunsu/Workspace/aia_pixel_dl/pointing_table.pkl")
    parser.add_argument("--do_fix_observer_location", type=bool, default=True)
    parser.add_argument("--do_respike", type=bool, default=False)
    parser.add_argument("--do_deconvolve", type=bool, default=True)
    parser.add_argument("--psfs", type=str, default="/Users/eunsu/Workspace/aia_pixel_dl/psfs.pkl")
    parser.add_argument("--do_correct_degradation", type=bool, default=True)
    parser.add_argument("--correction_table", type=str, default="/Users/eunsu/Workspace/aia_pixel_dl/correction_table.pkl")
    args = parser.parse_args()

    if args.pointing_table is not None :
        with open(args.pointing_table, 'rb') as f :
            pointing_table = pickle.load(f)
    else :
        pointing_table = None

    if args.psfs is not None :
        with open(args.psfs, 'rb') as f :
            psfs = pickle.load(f)
    else :
        psfs = None

    if args.correction_table is not None :
        with open(args.correction_table, 'rb') as f :
            correction_table = pickle.load(f)
    else :
        correction_table = None

    from glob import glob
    list_fits = glob()

