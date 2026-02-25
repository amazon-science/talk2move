import torch
from PIL import Image
import numpy as np
from diffusers import QwenImageEditPipeline
from flow_grpo.diffusers_patch.qwenimage_edit_pipeline_with_logprob import pipeline_with_logprob
import importlib
import os

from tqdm import tqdm
import json

def load_json(path):
    """Load JSON file to Python object (dict/list)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

input_path =  '/path/to/data'
prompt_path = '/path/to/data'


model_id = "Qwen/Qwen-Image-Edit"
device = "cuda"

pipeline = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipeline = pipeline.to(device)

n_step = 100
checkpoint_path = f'/path/to/data'
pipeline.load_lora_weights(checkpoint_path, prefix=None)

output_path = f'outputs_flux_indoor_v2_mini_qwen_sft'
os.makedirs(output_path, exist_ok=True)
samplenames = os.listdir(input_path)

generator = torch.Generator()
generator.manual_seed(42)

for sname in tqdm(samplenames):
    if 'txt' in sname:
        continue
    if '_3' in sname or '_2' in sname:
        continue
    if os.path.exists(os.path.join(output_path,sname)):
        continue
    imgp = os.path.join(input_path, sname)
    promptp = os.path.join(prompt_path, sname.replace('png','json'))

    image = Image.open(imgp)
    image = [image]
    metadata = load_json(promptp)
    prompt = metadata['prompt']
    prompt = [prompt]
    negative_prompt = [" "]*len(prompt)
    width, height = 1024, 1024
    images = pipeline_with_logprob(pipeline,image,prompt,negative_prompt=negative_prompt, num_inference_steps=10,true_cfg_scale=4,sde_window_size=2,height=height,width=width,generator=generator,noise_level=0)

    for i, img in enumerate(images['images']):
        img.save(os.path.join(output_path,sname.replace('.png', f'_{i}.png'))) 
  
