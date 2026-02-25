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
# from flow_grpo.clip_score import CLIPScore

from lang_sam import LangSAM

def erode_mask(m, k=3, iters=1):  # Small erosion to reduce boundary contamination
    for _ in range(iters):
        m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=k//2)
    return m


class LPIPSScorer:
    def __init__(self, device):
        self.device = device
        self.seg_model = LangSAM()
        # self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        # self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
        # self.clip_model = CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14")
        # x, y: [B,3,H,W], can be in [0,1], we let LPIPS normalize itself
        # mask: [B,1,H,W] takes {0,1}
        self.loss_fn = lpips.LPIPS(net='vgg')     # 'alex' is faster, 'vgg' is more stable
        self.loss_fn.eval()

    # def parse_caption_txt(self, caption_path):
    #     with open(caption_path, 'r', encoding='utf-8') as f:
    #         batch_caption = f.read()
    #     prompt = batch_caption.split('Move ')[-1].split(' to ')[0].split('forward')[0] + ' .'
    #     instruction = batch_caption.split(',')[0].split(' to ')[-1]
    #     return prompt, instruction, batch_caption


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

    # def relative_position(self, box_a, box_b):
    #     boxes = np.array([box_a, box_b])[:, :4].reshape(2, 2, 2)
    #     center_a, center_b = boxes.mean(axis=-2)
    #     dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    #     offset = center_a - center_b
    #     revised_offset = np.maximum(np.abs(offset) - self.POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    #     if np.all(np.abs(revised_offset) < 1e-3):
    #         return 0
    #     dx, dy = revised_offset / np.linalg.norm(offset)
        
    #     return (np.abs(dx) + np.abs(dy))*0.5

    # def l1_distance(self, image0, image1):
    #     image0 = image0.convert('RGB')
    #     image1 = image1.convert('RGB')
    #     image0 = image0.resize(image1.size)
    #     image0 = np.array(image0)
    #     image1 = np.array(image1)
    #     l1_distance = np.sum(np.abs(image0 - image1))
    #     num_pixels = image0.shape[0] * image0.shape[1] * image0.shape[2]
    #     normalized_l1_distance = l1_distance / num_pixels / 255
    #     return normalized_l1_distance        
        
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
                


       
