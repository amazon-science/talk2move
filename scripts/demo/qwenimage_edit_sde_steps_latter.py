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
device = "cuda"

# pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# pipe = pipe.to(device)

from tqdm import tqdm
import os
import json
from concurrent import futures

def load_json(path):
    """Load JSON file to Python object (dict/list)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def plot_distribution(data_list, title="Reward Distribution", xlabel="Value", ylabel="Frequency", bins=20, save_path=None):
    """
    Plot distribution histogram of 0~1 data, with optional save to image

    Parameters:
        data_list (list of float): Input data, typically numbers between 0 and 1
        title (str): Title of the plot
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        bins (int): Number of bins, default 20
        save_path (str or None): If path is provided, save plot to that path
    """
    plt.figure(figsize=(8, 4))
    plt.hist(data_list, bins=bins, range=(0, 1), color='skyblue', edgecolor='black')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Distribution plot saved to: {save_path}")

# input_path =  '/path/to/data'
# prompt_path = '/path/to/data'


input_path =  '/path/to/data'
prompt_path = '/path/to/data'

config_reward_fn = {
        "manipulation": 1.0,
        # "ours_clip": 1.0,
        # "rotation": 1.0
        # "editscore": 1.0
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
        default=512,
        help="Number of rollouts (default: 16)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
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


for n_step in [100,140,120]:

    counter = 0
    with torch.inference_mode():
        # Load the pipeline
        pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
        pipeline.to(torch.bfloat16)
        pipeline.to("cuda")

        # Load trained LoRA weights for in-scene editing
        # checkpoint_path = f"/path/to/data"
        checkpoint_path = f'/path/to/data'
        pipeline.load_lora_weights(checkpoint_path, prefix=None)

        output_path = f'preprocess_demos_step{n_step}/outputs_sde_step{num_step}'
        os.makedirs(output_path, exist_ok=True)
        ff = open(f'preprocess_demos_step{n_step}/log_step{num_step}.txt','w')

        samplenames = os.listdir(input_path)
        samplenames.sort()
        # print(samplenames[:10])
        total_rewards = {}
        for sname in tqdm(['fluxdata_indoor_v200006_2.png']):
            # if os.path.exists(os.path.join(output_path,sname)):
            #     continue
            imgp = os.path.join(input_path, sname)
            promptp = os.path.join(prompt_path, sname.replace('.png','.json'))
            # f = open(promptp, 'r')
            # prompt = f.readline()
            # f.close()
            # if not os.path.exists(promptp):
            #     continue
            ref_image = Image.open(imgp)

            metadata = load_json(promptp)
            print(metadata)
            if 'direction' not in list(metadata.keys()):
                continue
            prompt = metadata['prompt']
            width, height = 512, 512
            ref_images = [ref_image]*batchsize
            prompts = [prompt]*batchsize
            negative_prompts = [" "]*len(prompts)
            metadatas = [metadata]*batchsize

            num_batches = num_rollouts // batchsize


            latent_height = 2 * (int(height) // (pipeline.vae_scale_factor * 2))
            latent_width = 2 * (int(width) // (pipeline.vae_scale_factor * 2))
            num_channels_latents = pipeline.transformer.config.in_channels // 4
            shape = (1, 1, num_channels_latents, latent_height, latent_width)
            rand_noise = randn_tensor(shape, generator=generator, device=pipeline.device, dtype=pipeline.dtype)
            rand_noise = torch.cat([rand_noise]*batchsize, dim=0)
            rand_noise = pipeline._pack_latents(rand_noise, batchsize, num_channels_latents, latent_height, latent_width)

            images_ode = pipeline_with_logprob(pipeline, ref_images[:1],prompts[:1],negative_prompt=negative_prompts[:1], \
                        num_inference_steps=10,true_cfg_scale=4,sde_window_size=-1,sde_window_range=(num_step,num_step+2),height=height,width=width,\
                        generator=generator,noise_level=0,process_index=0,latents=rand_noise[:1])
            for j, img in enumerate(images_ode['images']):
                img.save(os.path.join(output_path,sname.replace('.png', f'_ode.png'))) 

            rewards_ode = executor.submit(reward_fn, images_ode['images'], prompts[:1], metadatas[:1], ref_images=ref_images[:1], only_strict=False)
            reward_odes, _ = rewards_ode.result()
            reward_ode = {}
            for key, value in reward_odes.items():
                if isinstance(value[0], torch.Tensor):
                    value = value[0].item()
                reward_ode[key] = value
            print(reward_ode)
            print(metadatas[0])
            # if reward_ode['avg'] > 0.1:
            #     continue
            # reward_ode = reward_ode["avg"][0].item()
            reward_sde = {}
            for i in range(num_batches):
                images = pipeline_with_logprob(pipeline,ref_images,prompts,negative_prompt=negative_prompts, \
                        num_inference_steps=10,true_cfg_scale=4,sde_window_size=num_step,sde_window_range=(0,num_step),height=height,width=width,\
                        generator=generator,noise_level=1.0,process_index=0,latents=rand_noise)

                for j, img in enumerate(images['images']):
                    img.save(os.path.join(output_path,sname.replace('.png', f'_{i*batchsize+j}.png'))) 

                rewards = executor.submit(reward_fn, images['images'], prompts, metadatas, ref_images=ref_images, only_strict=False)
                # yield to to make sure reward computation starts

                rewards, reward_metadata = rewards.result()
                
                for key, value in rewards.items():
                    if key not in total_rewards.keys():
                        total_rewards[key] = []
                    if key not in reward_sde.keys():
                        reward_sde[key] = []
                    for reward in rewards[key]:
                        print(key, reward)
                        if isinstance(reward, torch.Tensor):
                            reward = reward.item()
                        total_rewards[key].append(reward)
                        reward_sde[key].append(reward)
                # for reward in rewards['avg']:
                #     total_rewards.append(reward.item())
                #     reward_sde.append(reward.item())
                # print(rewards)
            f = open(f'preprocess_demos_step{n_step}/step{num_step}_{sname[:-4]}.txt', 'w')
            # print(reward_sde)
            for key in reward_sde.keys():
                fig_save_path = os.path.join(f'preprocess_demos_step{n_step}', f'step{num_step}_{key}_reward_hist_{sname}')
                
                plot_distribution(reward_sde[key], save_path=fig_save_path)
        
                for rew in reward_sde[key]:
                    f.write(f'{key}_{rew}\n')
            

                reward_sde_per = np.array(reward_sde[key])
                avg_dist =np.mean((reward_sde_per - reward_ode[key]) * (reward_sde_per - reward_ode[key]) )

                print('-------------',sname, np.mean(reward_sde_per), np.std(reward_sde_per), avg_dist)
                ff.write(f'{key} {sname}, {np.mean(reward_sde_per)}, {np.std(reward_sde_per)}, {avg_dist}, {reward_ode[key]}\n')
            
            f.close()
            counter += 1
            if counter == 4:
                break

    for key, value in total_rewards.items():      
        print(key, np.mean(value), np.std(value))
    ff.close()