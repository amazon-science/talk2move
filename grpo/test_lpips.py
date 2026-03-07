import torch
import torch.nn.functional as F
import lpips

import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from grpo.clip_score import CLIPScore

from lang_sam import LangSAM
seg_model = LangSAM()

def erode_mask(m, k=3, iters=1):  # Small erosion to reduce boundary contamination
    for _ in range(iters):
        m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=k//2)
    return m

# x, y: [B,3,H,W], can be in [0,1], we let LPIPS normalize itself
# mask: [B,1,H,W] takes {0,1}
loss_fn = lpips.LPIPS(net='vgg')     # 'alex' is faster, 'vgg' is more stable
loss_fn.eval()

@torch.no_grad()
def masked_lpips_replace_outside(x, y, mask, erode_iters=1):
    mask, x, y = torch.tensor(mask), torch.tensor(x), torch.tensor(y)
    m = mask.float().clamp(0,1)
    for _ in range(erode_iters):
        m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
    y2 = y * m + x * (1 - m)     # Replace outside mask with same as x
    score = loss_fn(x, y2, normalize=True)  # [B,1,1,1]
    return score.view(-1)        # Smaller is more similar

ref_path =  '/path/to/data'
# '/path/to/data'
tgt_path = '/path/to/data'
prompt = 'carpet. '
ref_image = Image.open(ref_path)
image = Image.open(tgt_path)
masks_input = seg_model.predict([ref_image], [prompt])
masks_output = seg_model.predict([image], [prompt])

image_h, image_w = ref_image.size
if (len(masks_input[0]['masks']) == 0) or (len(masks_output[0]['masks']) == 0): 
    source_image_crop, edit_image_crop = None, None
    
else:
    boxes_input = masks_input[0]['boxes'][0]
    boxes_output = masks_output[0]['boxes'][0]
    image1 = ref_image
    image2 = image

    mask1 = masks_input[0]['masks'][0].reshape((image_h, image_w))
    mask2 = masks_output[0]['masks'][0].reshape((image_h, image_w))
    mask = mask1.astype(bool) | mask2.astype(bool)
    mask = mask.reshape((1,1, 1024, 1024))
    
    image1 = np.array(image1).transpose(2,0,1)
    image2 = np.array(image2).transpose(2,0,1)
    score = masked_lpips_replace_outside(image1, image2, ~mask)
    print(score)