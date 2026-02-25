from paths import *
from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
import torch
from PIL import Image

import torch.nn.functional as F
from utils import *
from inference import *

import time

from huggingface_hub import hf_hub_download
# ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="croplargeEX2/dino_weight.pt", repo_type="model", cache_dir='./', resume_download=True)
# print(ckpt_path)
ckpt_path = '/path/to/data'
save_path = './'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dino = DINOv2_MLP(
                    dino_mode   = 'large',
                    in_dim      = 1024,
                    out_dim     = 360+180+180+2,
                    evaluate    = True,
                    mask_dino   = False,
                    frozen_back = False
                )

dino.eval()
print('model create')
dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
dino = dino.to(device)
print('weight loaded')
val_preprocess   = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')

img1 = "/path/to/data"
img2 = "/path/to/data"
start_time = time.time()
for image_path in [img1, img2]:
    origin_image = Image.open(image_path).convert('RGB')
    angles = get_3angle(origin_image, dino, val_preprocess, device)
    azimuth     = float(angles[0])
    polar       = float(angles[1])
    rotation    = float(angles[2])
    confidence  = float(angles[3])
    print(azimuth, polar, rotation)
    print(time.time()-start_time)
    start_time = time.time()
