import os

import cv2
import numpy as np
import torch
from PIL import Image
import glob
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
root_dir = "/home/jz927/Documents/relighting/neuralgaffer/jinx-synthetic/light"

def read_hdr(path):
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    try:
        with open(path, "rb") as h:
            buffer_ = np.frombuffer(h.read(), np.uint8)
        bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error reading HDR file {path}: {e}")
        rgb = None
        return rgb
    rgb = torch.from_numpy(rgb)
    return rgb

def get_envir_map_light(envir_map, incident_dir):

    envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
    phi = torch.arccos(incident_dir[:, 2]).reshape(-1) - 1e-6
    theta = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1)
    # normalize to [-1, 1]
    query_y = (phi / np.pi) * 2 - 1
    query_x = - theta / np.pi
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
    light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

    return light_rgbs

all_hdr_dirs = glob.glob(os.path.join(root_dir, "*"))
print(all_hdr_dirs)
# exit(0)
for hdr_dir in all_hdr_dirs:
    all_dir = os.path.join(hdr_dir, "hdr")
    exr_images = glob.glob(os.path.join(all_dir, "*"))
    # print(exr_images)
    exr_images = sorted(
        os.path.basename(fname) for fname in exr_images)# if os.path.splitext(fname)[1].lower() == "exr")
    data = exr_images
    # print(data)
    # exit(0)

    ldr_dir = os.path.join(hdr_dir, "LDR")
    hdr_dir = os.path.join(hdr_dir, "HDR_normalized")
    # background_dir = "../jewelry/jinx-synthetic/engagement-ring/background"
    # print("1")
    # if not os.path.exists(background_dir):
        # os.makedirs(background_dir)
    if not os.path.exists(ldr_dir):
        os.makedirs(ldr_dir)
    if not os.path.exists(hdr_dir):
        os.makedirs(hdr_dir)
    # print("2")
    # hdr_rgb = read_hdr("11_vignaioli_night_4k.exr")
    # print(data)
    for d in data:
        print(d)
        file_name = Path(d).stem
        print(file_name)
        hdr_rgb = read_hdr(os.path.join(all_dir, d))
        print(f"hdr_rgb: {hdr_rgb.shape} min {hdr_rgb.min()} max {hdr_rgb.max()}")
        # envir_map_results = get_envir_map_light(hdr_rgb, view_dirs_world).clamp(0, 1)
        # envir_map_results = hdr_rgb ** (1/2.2)
        # # print(envir_map_results.shape)
        # envir_map_results = envir_map_results.reshape(256, 256, 3)
        # envir_map_results = np.uint8(envir_map_results * 255)
        # cur_results.append(envir_map_results.copy())
        #         # torch to Image
        # envir_map_results = Image.fromarray(envir_map_results)
        # envir_map_results.save(os.path.join(background_dir, file_name, ".png"))

        

        ldr = hdr_rgb.clamp(0, 1) ** (1 / 2.2)
        hdr = torch.log1p(10 * hdr_rgb)
        hdr = hdr / hdr.max()  # rescale to [0, 1]

        ldr = (ldr.cpu().numpy() * 255.0).astype(np.uint8)
        hdr = (hdr.cpu().numpy() * 255.0).astype(np.uint8)

        Image.fromarray(ldr).save(os.path.join(ldr_dir, f'{file_name}.png'))
        Image.fromarray(hdr).save(os.path.join(hdr_dir, f'{file_name}.png'))
