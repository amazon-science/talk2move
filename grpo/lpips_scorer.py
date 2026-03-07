import os
import json
import torch
import torch.nn.functional as F
import lpips

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from lang_sam import LangSAM

def erode_mask(m, k=3, iters=1):  # Small erosion to reduce boundary contamination
    for _ in range(iters):
        m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=k//2)
    return m


class LPIPSScorer:
    def __init__(self, device):
        self.device = device
        self.seg_model = LangSAM()

        self.loss_fn = lpips.LPIPS(net='vgg')     # 'alex' is faster, 'vgg' is more stable
        self.loss_fn.eval()


    @torch.no_grad()
    def masked_lpips_replace_outside(self, x, y, mask, erode_iters=1):
        mask, x, y = torch.tensor(mask), torch.tensor(x), torch.tensor(y)
        m = mask.float().clamp(0,1)
        for _ in range(erode_iters):
            m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
        y2 = y * m + x * (1 - m)     # Replace outside mask with same as x
        score = self.loss_fn(x, y2, normalize=True)  # [B,1,1,1]
        return score.view(-1)        # Smaller is more similar

    def object_detection(self, image, prompt):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.4,
            target_sizes=[image.size[::-1]]
        )
        return [box.tolist() for box in results[0]["boxes"]] if results else []

        
    @torch.no_grad()
    def run(self, images, ref_images, prompts, metadatas):
        scores = []
        for image, ref_image, prompt, metadata in zip(images, ref_images, prompts, metadatas):
            correct, error_msg = 0, ""
            objectname = metadata['object']
            # objectname = [objectname+'. ']
            objectname = objectname+'. '
            # boxes_input = self.object_detection(ref_image, [objectname]) 
            # boxes_output = self.object_detection(image, [objectname])
            ref_image = ref_image.resize(image.size)
            image_h, image_w = image.size

            masks_input = self.seg_model.predict([ref_image], [objectname])
            masks_output = self.seg_model.predict([image], [objectname])
            
            direction = metadata['direction'] 

            if len(masks_input[0]['masks']) == 0: 
                score = 0
            elif len(masks_output[0]['masks']) == 0:
                # axis_correct, angle_correct, direction_correct = 0, 0, 0
                score = 0
            else:
                mask1 = masks_input[0]['masks'][0].reshape((image_h, image_w))
                mask2 = masks_output[0]['masks'][0].reshape((image_h, image_w))
                mask = mask1.astype(bool) | mask2.astype(bool)
                mask = mask.reshape((1,1,image_h,image_w))
                
                image1 = np.array(ref_image).transpose(2,0,1)
                image2 = np.array(image).transpose(2,0,1)
                score = 1 - self.masked_lpips_replace_outside(image1, image2, ~mask)
                # print(score)   

                            
            total_score = score
            scores.append(total_score)
        scores = torch.tensor(scores, device=self.device)
        return scores
                


       
