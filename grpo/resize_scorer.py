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


class ResizeScorer:
    def __init__(self, device):
        self.device = device
        self.seg_model = LangSAM()
        # self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        # self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
        # self.clip_model = CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14")
        self.POSITION_THRESHOLD = 0.1

    # def parse_caption_txt(self, caption_path):
    #     with open(caption_path, 'r', encoding='utf-8') as f:
    #         batch_caption = f.read()
    #     prompt = batch_caption.split('Move ')[-1].split(' to ')[0].split('forward')[0] + ' .'
    #     instruction = batch_caption.split(',')[0].split(' to ')[-1]
    #     return prompt, instruction, batch_caption

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

    def relative_position(self, box_a, box_b):
        boxes = np.array([box_a, box_b])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = np.maximum(np.abs(offset) - self.POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return 0
        dx, dy = revised_offset / np.linalg.norm(offset)
        
        return (np.abs(dx) + np.abs(dy))*0.5

    def l1_distance(self, image0, image1):
        image0 = image0.convert('RGB')
        image1 = image1.convert('RGB')
        image0 = image0.resize(image1.size)
        image0 = np.array(image0)
        image1 = np.array(image1)
        l1_distance = np.sum(np.abs(image0 - image1))
        num_pixels = image0.shape[0] * image0.shape[1] * image0.shape[2]
        normalized_l1_distance = l1_distance / num_pixels / 255
        return normalized_l1_distance        
        
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

            masks_input = self.seg_model.predict([ref_image], [objectname])
            masks_output = self.seg_model.predict([image], [objectname])

            # direction = metadata['direction'] 

            if len(masks_input[0]['masks']) == 0: 
                correct = 0
            elif len(masks_output[0]['masks']) == 0:
                # axis_correct, angle_correct, direction_correct = 0, 0, 0
                correct = 0
            else:
                boxes_input = masks_input[0]['boxes'][0]
                boxes_output = masks_output[0]['boxes'][0]
                x1_min, y1_min, x1_max, y1_max = boxes_input
                x2_min, y2_min, x2_max, y2_max = boxes_output
                w1, h1 = x1_max - x1_min, y1_max - y1_min
                w2, h2 = x2_max - x2_min, y2_max - y2_min
                x_c1, y_c1 = (x1_max + x1_min) / (w1+w2), (y1_max + y1_min) / (h1+h2)
                x_c2, y_c2 = (x2_max + x2_min) / (w1+w2), (y2_max + y2_min) / (h1+h2)
                # if np.linalg.norm((x_c2-x_c1, y_c2-y_c1)) > 0.2:
                #     correct = -1
                
                if 'bigger' in prompt:
                    w_ratio = w2 / w1
                    h_ratio = h2 / h1
                    # correct = 1.0
                    correct = (np.abs(w_ratio - metadata['ratio']) + np.abs(h_ratio - metadata['ratio'])) / 4# /  (2 * metadata['ratio'])
                    correct = 1 - correct

                elif 'smaller' in prompt:
                    w_ratio = w1 / w2
                    h_ratio = h1 / h2
                    # correct = 1.0
                    correct = (np.abs(w_ratio - metadata['ratio']) + np.abs(h_ratio - metadata['ratio'])) / 4#/  (2 * metadata['ratio'])
                    correct = 1 - correct
                else:
                    w_ratio = 0
                    h_ratio = 0
                    correct = 0
                    
                            
            total_score = correct
            scores.append(total_score)
        scores = torch.tensor(scores, device=self.device)
        return scores
                


       
