import torch
import torch.nn as nn
import torchvision
from transformers import AutoModel, AutoProcessor
from transformers import SiglipModel
from huggingface_hub import PyTorchModelHubMixin

import torchvision.transforms as T
from transformers import AutoImageProcessor

from PIL import Image
import numpy as np
from einops import rearrange

def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")

def get_image_transform(processor:AutoImageProcessor):
    config = processor.to_dict()
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()

    return T.Compose([resize, crop, normalise])

class CLIPScore(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="imscore",
    repo_url="https://github.com/RE-N-Y/imscore"
):
    def __init__(self, tag:str):
        super().__init__()
        self.model = AutoModel.from_pretrained(tag)
        self.processor = AutoProcessor.from_pretrained(tag)
        self.tform = get_image_transform(self.processor.image_processor)

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs.logits_per_image
    
    def score(self, image1, image2):

        inputs1 = self.processor(images=image1, return_tensors="pt")
        image1_features = self.model.get_image_features(**inputs1)
        inputs2 = self.processor(images=image2, return_tensors="pt")
        image2_features = self.model.get_image_features(**inputs2)
        # print(image1_features.shape, image1_features.shape)
        # Step 4: Normalize & compute cosine similarity
        image1_features /= image1_features.norm(dim=-1, keepdim=True)
        image2_features /= image2_features.norm(dim=-1, keepdim=True)

        similarity = (image1_features @ image2_features.T).item()

        return similarity

