import torch
from PIL import Image
import flow_grpo.rewards
import argparse

import numpy as np
from diffusers import QwenImageEditPipeline
from diffusers.utils.torch_utils import randn_tensor
from flow_grpo.diffusers_patch.qwenimage_edit_pipeline_with_logprob import pipeline_with_logprob
import importlib
import matplotlib.pyplot as plt

model_id = "Qwen/Qwen-Image-Edit"


from tqdm import tqdm
import os
import json
from concurrent import futures

def load_json(path):
    """Load JSON file to Python object (dict/list)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



input_path =  '/path/to/data'
prompt_path = '/path/to/data'
device = 'cuda'
config_reward_fn = {
        "manipulation": 1.0,
    }

reward_fn = getattr(flow_grpo.rewards, 'multi_score')(device, config_reward_fn)

def parse_args():
    parser = argparse.ArgumentParser(description="QwenImage Edit SDE Demo")
    parser.add_argument(
        "--num_step", 
        type=int, 
        default=0, 
        help="Number of steps for SDE window range (default: 0)"
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="Number of rollouts (default: 16)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    return parser.parse_args()

args = parse_args()

executor = futures.ThreadPoolExecutor(max_workers=8)

num_step = args.num_step
num_rollouts = args.num_rollouts
batchsize = args.batch_size

generator = torch.Generator()
generator.manual_seed(args.seed)


with torch.inference_mode():
    # Load the pipeline
    pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda")

    checkpoint_path = f'/path/to/data'
    pipeline.load_lora_weights(checkpoint_path, prefix=None)

    output_path = f'original_ode/'
    os.makedirs(output_path, exist_ok=True)
    
    samplenames = os.listdir(input_path)
    samplenames.sort()
    # print(samplenames[:10])
    total_rewards = []
    for sname in tqdm(samplenames[:5]):
        imgp = os.path.join(input_path, sname)
        promptp = os.path.join(prompt_path, sname.replace('.png','_0.json'))
        ref_image = Image.open(imgp)
        metadata = load_json(promptp)
        prompt = metadata['prompt']
        width, height = 1024, 1024
        ref_images = [ref_image]*batchsize
        prompts = [prompt]*batchsize
        negative_prompts = [" "]*len(prompts)
        metadatas = [metadata]*batchsize

        num_batches = num_rollouts // batchsize


        latent_height = 2 * (int(height) // (pipeline.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (pipeline.vae_scale_factor * 2))
        num_channels_latents = pipeline.transformer.config.in_channels // 4
        shape = (1, 1, num_channels_latents, latent_height, latent_width)
        # rand_noise = randn_tensor(shape, generator=generator, device=pipeline.device, dtype=pipeline.dtype)
        rand_noise = torch.load('init_rand_noise')
        rand_noise = torch.cat([rand_noise]*batchsize, dim=0)
        rand_noise = pipeline._pack_latents(rand_noise, batchsize, num_channels_latents, latent_height, latent_width)
        for i in range(num_batches):
            # for sname in tqdm(samplenames[:30]): 
            inputs = {
                "image": ref_image,
                "prompt": prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 16,
                "latents":rand_noise
            }

            with torch.inference_mode():
                output = pipeline(**inputs)
                output_image = output.images[0]

                output_image.save(os.path.join(output_path,sname.replace('.png',f'.png')))
