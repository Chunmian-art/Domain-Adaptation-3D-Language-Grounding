import gzip
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import clip
import torch

shapenet_images_path = "shapenet-images/screenshots/"

data_np = "raw_np/"
keys = os.listdir(shapenet_images_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
with torch.no_grad():
    for key in tqdm(keys):
        pngs = os.listdir(os.path.join(shapenet_images_path, f"{key}"))
        pngs = [os.path.join(shapenet_images_path, f"{key}", p) for p in pngs if "png" in p]
        pngs.sort()
        for png in pngs:
            image = Image.open(png)
            image = preprocess(image).numpy()
            name = png.split('/')[-1].replace(".png", "")
            np.save(data_np+name+".npz", image)
        # data_np[name] = image
        # del image_features
# save_path_np = './data/shapenet-np.json.gz'
# json.dump(data_np, gzip.open(save_path_np, 'wt'))
