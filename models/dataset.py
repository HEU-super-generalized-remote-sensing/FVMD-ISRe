import os
import utm
import json
import rpcm
import torch
import numpy as np
import models.sat_utils as sat_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.train = True
        self.img_downscale = conf.get_float('img_downscale')
        self.data_dir = conf.get_string('data_dir')

        with open(os.path.join(self.data_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        self.json_files = [os.path.join(self.data_dir, json_p) for json_p in json_files]

        all_rgbs, all_rays, all_sun_dirs, all_ids = [], [], [], []
        for t, json_p in enumerate(self.json_files):
            # read json, image path and id
            with open(json_p) as f:
                d = json.load(f)
            img_p = os.path.join(self.data_dir, d["img"])
            img_id = os.path.splitext(os.path.basename(d["img"]))[0]

            # get rgb colors
            rgbs = sat_utils.load_tensor_from_rgb_geotiff(img_p, self.img_downscale)

            # get rays
            cache_path = "{}/{}.data".format(self.data_dir, img_id)
            if os.path.exists(cache_path) and self.conf.get_bool('force_reload'):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                rays = self.get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
                torch.save(rays, cache_path)

            # get sun direction
            sun_dirs = sat_utils.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])

            all_ids += [t * torch.ones(rays.shape[0], 1)]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]
            
            print("Image {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))
        
        self.all_ids = torch.cat(all_ids, 0).cpu()
        self.all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        self.all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        self.all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        self.cal_scaling_params(self.all_rays)
        self.all_rays = self.normalize_rays(self.all_rays)
        self.all_rays = torch.hstack([self.all_rays, self.all_sun_dirs])  # (len(json_files)*h*w, 11)
        
        self.val_range = None
        with open(os.path.join(self.data_dir, "{}_DSM.txt".format(img_id[:7])), "r") as f:
            val_data = f.read().split("\n")
        self.val_range = torch.tensor([[float(val_data[0]), float(val_data[1])],
                                       [float(val_data[0]) + float(val_data[2]) * float(val_data[3]), float(val_data[1]) + float(val_data[2]) * float(val_data[3])]]).cpu()
        self.val_range[:, 0] -= self.center[0]
        self.val_range[:, 1] -= self.center[1]
        self.val_range = torch.cat((self.val_range, torch.tensor([float(d["min_alt"]) - self.center[2], float(d["max_alt"]) - self.center[2]]).unsqueeze(-1).cpu()), -1)

        self.val_range[:, 0] /= self.range
        self.val_range[:, 1] /= self.range
        self.val_range[:, 2] /= self.range

        print('Load data: End')

    def cal_scaling_params(self, all_rays):
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = sat_utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = sat_utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = sat_utils.rpc_scaling_alt_params(all_points[:, 2])
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])]).cpu()
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])])).cpu()

    def normalize_rays(self, rays):
        rays[:, 0] -= self.center[0]
        rays[:, 1] -= self.center[1]
        rays[:, 2] -= self.center[2]
        rays[:, 0] /= self.range
        rays[:, 1] /= self.range
        rays[:, 2] /= self.range
        rays[:, 6] /= self.range
        rays[:, 7] /= self.range
        return rays

    def get_rays(self, cols, rows, rpc, min_alt, max_alt):
        min_alts = float(min_alt) * np.ones(cols.shape)
        max_alts = float(max_alt) * np.ones(cols.shape)
        # assume the points of maximum altitude are those closest to the camera
        lons, lats = rpc.localization(cols, rows, max_alts)
        # x_near, y_near, z_near = sat_utils.latlon_to_ecef_custom(lats, lons, max_alts)
        x_near, y_near, _, _ = utm.from_latlon(lats, lons, 15, 'T')
        xyz_near = np.vstack([x_near, y_near, max_alts]).T

        # similarly, the points of minimum altitude are the furthest away from the camera
        lons, lats = rpc.localization(cols, rows, min_alts)
        # x_far, y_far, z_far = sat_utils.latlon_to_ecef_custom(lats, lons, min_alts)
        x_far, y_far, _, _ = utm.from_latlon(lats, lons, 15, 'T')
        xyz_far = np.vstack([x_far, y_far, min_alts]).T

        # define the rays origin as the nearest point coordinates
        rays_o = xyz_near

        # define the unit direction vector
        d = xyz_far - xyz_near
        rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

        # assume the nearest points are at distance 0 from the camera
        # the furthest points are at distance Euclidean distance(far - near)
        fars = np.linalg.norm(d, axis=1)
        nears = float(0) * np.ones(fars.shape)

        # create a stack with the rays origin, direction vector and near-far bounds
        rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
        rays = rays.type(torch.FloatTensor)
        return rays
    
    def __len__(self):
        return self.all_rays.shape[0]

    def __getitem__(self, idx):
        if self.train:
            return (self.all_rays[idx], self.all_rgbs[idx])
        else:
            with open(self.json_files[idx]) as f:
                d = json.load(f)
            return (self.all_rays[torch.where(self.all_ids == idx)[0]], int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale))
