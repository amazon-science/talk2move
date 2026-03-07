import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from grpo.clip_score import CLIPScore

class translationScorer:
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
        self.clip_model = CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14")
        self.POSITION_THRESHOLD = 0.1

    def parse_caption_txt(self, caption_path):
        with open(caption_path, 'r', encoding='utf-8') as f:
            batch_caption = f.read()
        prompt = batch_caption.split('Move ')[-1].split(' to ')[0].split('forward')[0] + ' .'
        instruction = batch_caption.split(',')[0].split(' to ')[-1]
        return prompt, instruction, batch_caption

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

    def relative_position(self, box_a, box_b, relation):
        boxes = np.array([box_a, box_b])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = np.maximum(np.abs(offset) - self.POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return False
        dx, dy = revised_offset / np.linalg.norm(offset)
        if (dx < -0.01 and 'right' in relation) or (dx > 0.01 and 'left' in relation) :
            return np.abs(dx)
        elif (dy < -0.01 and 'down' in relation) or (dy > 0.01 and 'up' in relation):
            return np.abs(dy)
        return 0

    def image_consistency(self, image0, image1):
        with torch.no_grad():
            return self.clip_model.score(image0, image1)
        
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
            sim_overall = self.image_consistency(ref_image, image)
            # sim_overall = int(sim_overall >= 0.9)
            l1_distance = self.l1_distance(ref_image, image)
            correct, error_msg = 0, ""
            objectname = metadata['object']
            objectname = [objectname+'. ']
            boxes_input = self.object_detection(ref_image, objectname)
            boxes_output = self.object_detection(image, objectname)
            sim_obj, sim_old = 0, 0
            if not boxes_input or not boxes_output:
                error_msg = "Missing boxes."
                correct = 0
            else:
                correct = self.relative_position(boxes_input[0], boxes_output[0], metadata['direction'])
                # sim_obj = self.image_consistency(
                #     ref_image.crop(boxes_input[0]), image.crop(boxes_output[0]))
                # sim_obj = int(sim_obj >= 0.75)
                # sim_old = self.image_consistency(
                #     ref_image.crop(boxes_input[0]), image.crop(boxes_input[0]))
                # sim_old = int(sim_old < 0.95)
            print('correct, sim_overall, l1', correct, sim_overall, l1_distance)
            total_score = correct * 0.8 + sim_overall * 0.1 #* (sim_overall + 0.5 * sim_obj + 0.5 * sim_old + (1 - l1_distance)) / 3
            scores.append(total_score)
        scores = torch.tensor(scores, device=self.device)
        return scores
                # if sim_obj < 0.75:
                #     correct, error_msg = 0, "Object mismatch."
                # if sim_old > 0.95 or result['Source Location Cleared'] == 0:
                #     correct, error_msg = 0, "Original object not removed."


       
