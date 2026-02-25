import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from flow_grpo.clip_score import CLIPScore
from flow_grpo.qwen_caller import qwenvl_detection 

class PositionScorer:
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
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
        
    @torch.no_grad()
    def run(self, images, ref_images, prompts, metadatas):
        scores = []
        for image, ref_image, prompt, metadata in zip(images, ref_images, prompts, metadatas):
            correct, error_msg = 0, ""
            objectname = metadata['object']
            # objectname = [objectname+'. ']
            objectname = objectname+'. '
            boxes_input = self.object_detection(ref_image, [objectname]) 
            boxes_output = self.object_detection(image, [objectname])

            sim_obj, sim_old = 0, 0
            if not boxes_input or not boxes_output:
                error_msg = "Missing boxes."
                correct = 0
            else:
                correct = 1 - self.relative_position(boxes_input[0], boxes_output[0])
            
            if 'translation' in metadata['tag']:
                correct = 1
            
            total_score = correct
            scores.append(total_score)
        scores = torch.tensor(scores, device=self.device)
        return scores
                


       
