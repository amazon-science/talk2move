import torch
from PIL import Image
import numpy as np
from diffusers import QwenImageEditPipeline
from flow_grpo.diffusers_patch.qwenimage_edit_pipeline_with_logprob import pipeline_with_logprob
import importlib

model_id = "Qwen/Qwen-Image-Edit"
device = "cuda"

# pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# pipe = pipe.to(device)

from diffusers import QwenImageEditPipeline
import torch
from PIL import Image
from tqdm import tqdm
import os
import json

def load_json(path):
    """Load JSON file to Python object (dict/list)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


input_path =  '/path/to/data'
prompt_path = '/path/to/data'

n_steps = ['100']
for n_step in n_steps:
    # Load the pipeline
    pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda")

    # Load trained LoRA weights for in-scene editing
    # checkpoint_path = f"/path/to/data"
    checkpoint_path = f'/path/to/data'
    pipeline.load_lora_weights(checkpoint_path, prefix=None)

    output_path = f'outputs_flux_indoor_v2_mini_qwen_grpo_v3.6_0_1_2_3_{n_step}_seed0_ode3'
    os.makedirs(output_path, exist_ok=True)
    samplenames = os.listdir(input_path)
    
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
        metadata = load_json(promptp)
        prompt = metadata['prompt']
        width, height = 1024, 1024
        inputs = {
                "image": image,
                "prompt": prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 10,
            }
        with torch.inference_mode():
            output = pipeline(**inputs)
            output_image = output.images[0]
        for i, img in enumerate(output.images):
            img.save(os.path.join(output_path,sname.replace('.png', f'_{i}.png'))) 
  