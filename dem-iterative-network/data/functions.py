import numpy as np
import torch


def normalize_euv_numpy(data):
    tmp = np.clip(data+1., a_min=1., a_max=None)
    tmp = np.log10(tmp)/2. - 1.
    return tmp


def normalize_euv_torch(data):
    tmp = torch.clip(data+1., min=1., max=None)
    tmp = torch.log10(tmp)/2. - 1.
    return tmp


def normalize_euv(data):
    if isinstance(data, np.ndarray):
        return normalize_euv_numpy(data)
    elif isinstance(data, torch.Tensor):
        return normalize_euv_torch(data)
    else:
        raise NotImplementedError(f"Unsupported data type: {type(data)}")


def denormalize_euv_numpy(data):
    tmp = np.clip(data+1., a_min=0., a_max=None) * 2.
    tmp = np.power(10., tmp) - 1.
    return tmp


def denormalize_euv_torch(data):
    tmp = torch.clip(data+1., min=0., max=None) * 2.
    tmp = torch.pow(10., tmp) - 1.
    return tmp


def denormalize_euv(data):
    if isinstance(data, np.ndarray):
        return denormalize_euv_numpy(data)
    elif isinstance(data, torch.Tensor):
        return denormalize_euv_torch(data)
    else:
        raise NotImplementedError(f"Unsupported data type: {type(data)}")


def normalize_dem_numpy(data):
#    tmp = np.clip(data+1., a_min=1., a_max=None)
#    tmp = np.log10(tmp)/20. - 1.
    tmp = np.sqrt(data/1e22) - 1.
    return tmp


def normalize_dem_torch(data):
#    tmp = torch.clip(data+1., min=1., max=None)
#    tmp = torch.log10(tmp)/20. - 1.
    tmp = torch.sqrt(data/1e22) - 1.
    return tmp


def normalize_dem(data):
    if isinstance(data, np.ndarray):
        return normalize_dem_numpy(data)
    elif isinstance(data, torch.Tensor):
        return normalize_dem_torch(data)
    else:
        raise NotImplementedError(f"Unsupported data type: {type(data)}")


def denormalize_dem_numpy(data):
#    tmp = np.clip(data+1., a_min=0., a_max=None)*20
#    tmp = np.power(10., tmp) - 1.
    tmp = np.clip(data+1., a_min=0, a_max=None)
    tmp = 1e22*tmp*tmp
    return tmp


def denormalize_dem_torch(data):
#    tmp = torch.clip(data+1., min=0., max=None)*20
#    tmp = torch.pow(10., tmp) - 1.
    tmp = torch.clip(data+1., min=0, max=None)
    tmp = 1e22*tmp*tmp
    return tmp


def denormalize_dem(data):
    if isinstance(data, np.ndarray):
        return denormalize_dem_numpy(data)
    elif isinstance(data, torch.Tensor):
        return denormalize_dem_torch(data)
    else:
        raise NotImplementedError(f"Unsupported data type: {type(data)}")