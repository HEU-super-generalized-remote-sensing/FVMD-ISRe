import copy
import torch
import rasterio
import numpy as np
from PIL import Image
from torchvision import transforms as T


def load_tensor_from_rgb_geotiff(img_path, downscale_factor=1.0, imethod=Image.BICUBIC):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))
        img = T.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(img))
        img = np.transpose(img.numpy(), (1, 2, 0))
    img = T.ToTensor()(img)  # (3, h, w)
    rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = rgbs.type(torch.FloatTensor)
    return rgbs

def rescale_rpc(rpc, alpha):
    rpc_scaled = copy.copy(rpc)
    rpc_scaled.row_scale *= float(alpha)
    rpc_scaled.col_scale *= float(alpha)
    rpc_scaled.row_offset *= float(alpha)
    rpc_scaled.col_offset *= float(alpha)
    return rpc_scaled

def rpc_scaling_params(v):
    """
    find the scale and offset of a vector
    """
    scale = (v.max() - v.min()) / 2
    offset = v.min() + scale
    return scale, offset

def rpc_scaling_alt_params(v):
    """
    find the scale and offset of a vector
    """
    scale = (v.max() - v.min()) / 2
    offset = v.mean()
    return scale, offset

def latlon_to_ecef_custom(lat, lon, alt):
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z

def get_sun_dirs(sun_elevation_deg, sun_azimuth_deg, n_rays):
    """
    Get sun direction vectors
    Args:
        sun_elevation_deg: float, sun elevation in  degrees
        sun_azimuth_deg: float, sun azimuth in degrees
        n_rays: number of rays affected by the same sun direction
    Returns:
        sun_d: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
    """
    sun_el = np.radians(sun_elevation_deg)
    sun_az = np.radians(sun_azimuth_deg)
    sun_d = -1 * np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
    sun_d = sun_d / np.linalg.norm(sun_d)
    sun_dirs = torch.from_numpy(np.tile(sun_d, (n_rays, 1)))
    sun_dirs = sun_dirs.type(torch.FloatTensor)
    return sun_dirs