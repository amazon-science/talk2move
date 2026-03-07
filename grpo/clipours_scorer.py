import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from grpo.clip_score import CLIPScore

class ClipOursScorer:
    def __init__(self, device):
        self.device = device
        self.clip_model = CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14")


    def image_consistency(self, image0, image1):
        with torch.no_grad():
            return self.clip_model.score(image0, image1)
        
    @torch.no_grad()
    def run(self, images, ref_images, prompts, metadatas):
        scores = []
        for image, ref_image, prompt, metadata in zip(images, ref_images, prompts, metadatas):
            sim_overall = self.image_consistency(ref_image, image)
            scores.append(sim_overall)
        scores = torch.tensor(scores, device=self.device)
        return scores


       
