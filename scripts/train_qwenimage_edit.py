from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from ml_collections import config_flags
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
import logging


# Setup logger
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from diffusers import DiffusionPipeline, QwenImageTransformer2DModel, QwenImageEditPipeline
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_shift, calculate_dimensions  


from grpo.fsdp_utils import FSDPConfig, fsdp_wrapper, init_distributed, save_fsdp_checkpoint, save_lora_checkpoint, OptimizerOffload
import numpy as np
import grpo.prompts
import grpo.rewards
from grpo.stat_tracking import PerPromptStatTracker
from grpo.diffusers_patch.qwenimage_edit_pipeline_with_logprob import pipeline_with_logprob

from grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from diffusers.utils import convert_unet_state_dict_to_peft

import random
from torch.utils.data import Dataset, DataLoader, Sampler
from grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

def gather_tensor(tensor, world_size, timeout_seconds=300):
    if world_size == 1:
        return tensor
    
    # Add debugging info
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"Rank {rank}: gather_tensor called with tensor shape {tensor.shape}, dtype {tensor.dtype}, device {tensor.device}")
    
    # Validate tensor before gathering
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
        print(f"Rank {rank}: Made tensor contiguous")
    
    # Ensure all processes reach this point
    if dist.is_initialized():
        print(f"Rank {rank}: Waiting at barrier before all_gather")
        try:
            dist.barrier()
            print(f"Rank {rank}: Passed barrier, proceeding to all_gather")
        except Exception as e:
            print(f"Rank {rank}: Barrier failed: {e}")
            raise
    
    # Broadcast tensor shape to ensure consistency across processes
    if dist.is_initialized():
        tensor_shape = torch.tensor(list(tensor.shape), dtype=torch.long, device=tensor.device)
        shape_list = [torch.zeros_like(tensor_shape) for _ in range(world_size)]
        dist.all_gather(shape_list, tensor_shape)
        
        # Check if all processes have the same tensor shape
        for i, shape in enumerate(shape_list):
            if not torch.equal(shape, tensor_shape):
                print(f"Rank {rank}: Shape mismatch detected! Rank {i} has shape {shape.tolist()}, expected {tensor_shape.tolist()}")
                # Return just the local tensor if shapes don't match
                return tensor
    
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    try:
        print(f"Rank {rank}: Starting all_gather operation")
        dist.all_gather(gather_list, tensor)
        print(f"Rank {rank}: all_gather completed successfully")
        return torch.cat(gather_list)
    except Exception as e:
        print(f"Rank {rank}: all_gather failed with error: {e}")
        print(f"Rank {rank}: Tensor info - shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        print(f"Rank {rank}: Falling back to local tensor only")
        # Return local tensor if all_gather fails
        return tensor

def set_seed(seed, device_specific=True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_specific and torch.cuda.is_available():
        # For device-specific seeding
        torch.cuda.manual_seed_all(seed + dist.get_rank() if dist.is_initialized() else seed)

class translationPromptImageDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        item = {
            "prompt": self.prompts[idx],
            "metadata": self.metadatas[idx]
        }
        # Assuming 'image' in metadata contains a path to the image file
        image_path = self.metadatas[idx]['image']
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{image_path}"
        image = Image.open(os.path.join(self.dataset, image_path)).convert('RGB')
        item["image"] = image
        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        prompt_with_image_paths = [example["prompt_with_image_path"] for example in examples]
        return prompts, metadatas, images, prompt_with_image_paths

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, config, rank):

    # print(sample['calculated_shape'], sample["image_latents"].shape)
    calculated_width, calculated_height = sample['calculated_shape'][:2]
    calculated_width, calculated_height = int(calculated_width.item()), int(calculated_height.item())

    # print('img shape 2', calculated_width, calculated_height, config.resolution, config.resolution)
    # img_shapes = [[(1, config.resolution // pipeline.vae_scale_factor // 2, config.resolution // pipeline.vae_scale_factor // 2)]] * len(sample["latents"][:, j])
    
    img_shapes = [
            [
                (1, config.resolution // pipeline.vae_scale_factor // 2, config.resolution // pipeline.vae_scale_factor // 2),
                (1, calculated_height // pipeline.vae_scale_factor // 2, calculated_width// pipeline.vae_scale_factor // 2),
            ]
        ] * len(sample["latents"][:, j])
    

    txt_seq_lens = sample["prompt_embeds_mask"].sum(dim=1).tolist()
    negative_txt_seq_lens = sample["negative_prompt_embeds_mask"].sum(dim=1).tolist()

    # Predict the noise residual
    # txt_seq_lens is the longest, sample["prompt_embeds_mask"] and sample["prompt_embeds"] may have unnecessary padding
    sample["prompt_embeds_mask"] = sample["prompt_embeds_mask"][:, :max(txt_seq_lens+negative_txt_seq_lens)]
    sample["negative_prompt_embeds_mask"] = sample["negative_prompt_embeds_mask"][:, :max(txt_seq_lens+negative_txt_seq_lens)]
    sample["prompt_embeds"] = sample["prompt_embeds"][:, :max(txt_seq_lens+negative_txt_seq_lens)]
    sample["negative_prompt_embeds"] = sample["negative_prompt_embeds"][:, :max(txt_seq_lens+negative_txt_seq_lens)]

    latent_model_input = sample["latents"][:, j]
    if sample["image_latents"] is not None:
        latent_model_input = torch.cat([latent_model_input, sample["image_latents"]],dim=1)


    noise_pred = transformer(
        hidden_states=torch.cat([latent_model_input, latent_model_input], dim=0),
        timestep=torch.cat([sample["timesteps"][:, j], sample["timesteps"][:, j]], dim=0) / 1000,
        guidance=None,
        encoder_hidden_states_mask=torch.cat([sample["prompt_embeds_mask"], sample["negative_prompt_embeds_mask"]], dim=0),
        encoder_hidden_states=torch.cat([sample["prompt_embeds"], sample["negative_prompt_embeds"]], dim=0),
        img_shapes=img_shapes*2,
        txt_seq_lens=txt_seq_lens+negative_txt_seq_lens,
    )[0]
    noise_pred = noise_pred[:, : sample["latents"][:, j].size(1)]
    noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)


    comb_pred = neg_noise_pred + config.sample.guidance_scale * (noise_pred - neg_noise_pred)

    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    noise_pred = comb_pred * (cond_norm / noise_norm)
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, config, rank, local_rank, world_size, device, global_step, reward_fn, executor, autocast, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)
    generator = torch.Generator()
    generator.manual_seed(0)
    latent_height = 2 * (int(config.resolution) // (pipeline.vae_scale_factor * 2))
    latent_width = 2 * (int(config.resolution) // (pipeline.vae_scale_factor * 2))
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    shape = (1, 1, num_channels_latents, latent_height, latent_width)
    rand_noise = randn_tensor(shape, generator=generator, device=pipeline.device, dtype=pipeline.dtype)
    # rand_noise = torch.load('/path/to/data')

    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=local_rank != 0,
            position=0,
        ):
        prompts, prompt_metadata, ref_images, _ = test_batch
        batchsize = len(prompts)
        rand_noise2 = torch.cat([rand_noise]*batchsize, dim=0)
        rand_noise2 = pipeline._pack_latents(rand_noise2, batchsize, num_channels_latents, latent_height, latent_width)

        # print('test batch',rank, local_rank, prompts)
        generator = torch.Generator()
        generator.manual_seed(0)
        rand_noise = randn_tensor(shape, generator=generator, device=pipeline.device, dtype=pipeline.dtype)
        rand_noise2 = torch.cat([rand_noise]*batchsize, dim=0)
        rand_noise2 = pipeline._pack_latents(rand_noise2, batchsize, num_channels_latents, latent_height, latent_width)

        with autocast():
            with torch.no_grad():
                
                collected_data = pipeline_with_logprob(
                        pipeline,
                        ref_images,
                        prompts,
                        negative_prompt=[" "]*len(prompts),
                        num_inference_steps=config.sample.eval_num_steps,
                        true_cfg_scale=config.sample.guidance_scale,
                        height=config.resolution,
                        width=config.resolution, 
                        generator=generator,
                        noise_level=0,
                        sde_window_size=-1,
                        latents=rand_noise2,
                )

        images = collected_data["images"]

        images2 = images
        save_dir = os.path.join(config.save_dir, f"samples-eval-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        for idx, img in enumerate(images2):
            name = prompt_metadata[idx]['image'].split('/')[-1]
            idxx = idx % 8
            img.save(f'{save_dir}/{idxx}_{name}')
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, ref_images=ref_images, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
 
        rewards, reward_metadata = rewards.result()

        # Ensure all processes have completed reward computation before gathering
        if dist.is_initialized():
            dist.barrier()

        # Standardize reward tensor shapes across all processes
        for key, value in rewards.items():
            tensor_value = torch.as_tensor(value, device=device, dtype=torch.float32).contiguous()
            # Ensure all processes have the same batch size by padding if necessary
            rewards_gather = gather_tensor(tensor_value.to(torch.float32), world_size).cpu().float().numpy()
            all_rewards[key].append(rewards_gather)

    images = [torch.tensor(np.asarray(image)/255) for image in images]
    images = torch.stack(images, dim=0)
    images = images.permute(0,3,1,2)
    last_batch_images_gather = gather_tensor(torch.as_tensor(images, device=device), world_size).cpu().float().numpy()
    last_batch_prompt_ids = pipeline.tokenizer(
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    last_batch_prompt_ids_gather = gather_tensor(last_batch_prompt_ids, world_size).cpu().float().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = gather_tensor(torch.as_tensor(value,  dtype=torch.float32,device=device).contiguous(), world_size).cpu().float().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if rank == 0:
        os.makedirs('wandb_submit_eval', exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(4, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                pil.save(os.path.join('wandb_submit_eval', f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            # for key, value in all_rewards.items():
            #     print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


def get_transformer_layer_cls():
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionBlock, Qwen2_5_VLDecoderLayer
    return {
        QwenImageTransformerBlock,
        # QwenImageResidualBlock,
        # QwenImageResample,
        # QwenImageResidualBlock,
        # QwenImageMidBlock,
        # QwenImageAttentionBlock,
        Qwen2_5_VLVisionBlock,
        Qwen2_5_VLDecoderLayer
        }

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # Initialize distributed training
    is_distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    
    # Debug distributed setup
    print(f"Rank {rank}/{world_size}: Distributed training initialized")
    print(f"Rank {rank}: Device: {device}, Local rank: {local_rank}")
    print(f"Rank {rank}: CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Rank {rank}: CUDA device count: {torch.cuda.device_count()}")
        print(f"Rank {rank}: Current CUDA device: {torch.cuda.current_device()}")
    
    # Test basic distributed communication
    if dist.is_initialized() and world_size > 1:
        test_tensor = torch.tensor([rank], dtype=torch.bfloat16, device=device)
        print(f"Rank {rank}: Testing basic all_reduce...")
        try:
            dist.all_reduce(test_tensor)
            print(f"Rank {rank}: Basic all_reduce successful, result: {test_tensor.item()}")
        except Exception as e:
            print(f"Rank {rank}: Basic all_reduce failed: {e}")
            raise
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    if config.sample.sde_window_size > 0:
        num_train_timesteps = config.sample.sde_window_size
    else:
        num_train_timesteps = config.sample.num_steps - 1

    # Create project directory
    project_dir = os.path.join(config.logdir, config.run_name)
    os.makedirs(project_dir, exist_ok=True)
    if rank == 0:
        wandb.init(
            project="grpo",
            # mode="disabled"
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load scheduler, tokenizer and models.
    print(config.pretrained.model)
    
    pipeline = DiffusionPipeline.from_pretrained(
        config.pretrained.model, torch_dtype=inference_dtype
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=local_rank != 0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    
    pipeline.transformer.to(device)
    if config.use_lora:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
            "img_mod.1",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
            "txt_mod.1",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        pipeline.transformer.add_adapter(transformer_lora_config)

        if len(config.train.lora_path)>10:
            # Set correct lora layers
            lora_state_dict = pipeline.lora_state_dict(config.train.lora_path)

            transformer_state_dict = {
                f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }

            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(pipeline.transformer, transformer_state_dict, low_cpu_mem_usage=True)
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

    
    transformer = pipeline.transformer

    # Setup FSDP configuration
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,  # Set to True if memory is limited
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=inference_dtype,
        use_activation_checkpointing=config.activation_checkpointing,
        use_device_mesh=False, 
    )
    # Wrap language model with FSDP
    transformer.cpu().to(dtype=torch.float32)
    transformer = fsdp_wrapper(transformer, fsdp_config, get_transformer_layer_cls)
    pipeline.transformer = transformer

    if config.train.beta > 0:
        transformer_ref = QwenImageTransformer2DModel.from_pretrained(
            config.pretrained.model,
            subfolder="transformer",
            torch_dtype=inference_dtype
        )
        transformer_ref.eval()
        transformer_ref.requires_grad_(False)
        transformer_ref.cpu().to(dtype=torch.float32)
        transformer_ref = fsdp_wrapper(transformer_ref, fsdp_config, get_transformer_layer_cls)

    pipeline.text_encoder.cpu().to(dtype=torch.float32)
    pipeline.text_encoder = fsdp_wrapper(pipeline.text_encoder, fsdp_config, get_transformer_layer_cls)

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # print('Trainable transformer trainable parameters',transformer_trainable_parameters)
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    # ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=device)
    ema = None
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    if config.fsdp_optimizer_offload:
        optimizer = OptimizerOffload(optimizer)
    
    train_dataset = translationPromptImageDataset(config.dataset, 'train')
    test_dataset = translationPromptImageDataset(config.dataset, 'test')
    train_sampler = DistributedKRepeatSampler( 
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=42
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=translationPromptImageDataset.collate_fn,
        # persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset, shuffle=False),
        batch_size=config.sample.test_batch_size,
        collate_fn=translationPromptImageDataset.collate_fn,
        num_workers=8,
    )

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    if config.mixed_precision == "fp16":
        autocast = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    elif config.mixed_precision == "bf16":
        autocast = lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext

    # FSDP doesn't need deepspeed configuration
    # prepare prompt and reward fn
    reward_fn = getattr(grpo.rewards, 'multi_score')(device, config.reward_fn)
    eval_reward_fn = getattr(grpo.rewards, 'multi_score')(device, config.reward_fn)
    
    # FSDP setup completed above
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * world_size
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * world_size
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        pipeline.text_encoder.eval()
        if epoch % config.save_freq == 0 and epoch > 0:
            # save_fsdp_checkpoint(config.save_dir, transformer, global_step, rank)
            save_lora_checkpoint(config.save_dir, transformer, global_step, rank, pipeline)
        if epoch % config.eval_freq == 0: # and epoch > 0:
            eval(pipeline, test_dataloader, config, rank, local_rank, world_size, device, global_step, eval_reward_fn, executor, autocast, ema, transformer_trainable_parameters)

        start_time = time.time()
        #################### SAMPLING ####################
        pipeline.transformer.eval()
        pipeline.text_encoder.eval()
        samples = []
        # sample
        if config.sample.same_latent:
            generator = torch.Generator()
            generator.manual_seed(0)
            latent_height = 2 * (int(config.resolution) // (pipeline.vae_scale_factor * 2))
            latent_width = 2 * (int(config.resolution) // (pipeline.vae_scale_factor * 2))
            num_channels_latents = pipeline.transformer.config.in_channels // 4
            shape = (1, 1, num_channels_latents, latent_height, latent_width)
            batchsize = config.sample.train_batch_size
            rand_noise = randn_tensor(shape, generator=generator, device=pipeline.device, dtype=pipeline.dtype)
            rand_noise = torch.cat([rand_noise]*batchsize, dim=0)
            rand_noise = pipeline._pack_latents(rand_noise, batchsize, num_channels_latents, latent_height, latent_width)
        else:
            generator = None
            rand_noise = None

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=local_rank != 0,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata, ref_images, prompt_with_image_paths = next(train_iter)
            prompt_ids = pipeline.tokenizer(
                prompt_with_image_paths,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            
            with autocast():
                with torch.no_grad():
                    # print(len(ref_images),len(prompts))
                    collected_data = pipeline_with_logprob(
                        pipeline,
                        ref_images,
                        prompts,
                        negative_prompt=[" "]*len(prompts),
                        num_inference_steps=config.sample.num_steps,
                        true_cfg_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        sde_window_size=config.sample.sde_window_size,
                        sde_window_range=config.sample.sde_window_range,
                        process_index=rank,
                        latents=rand_noise
                    )

            latents = torch.stack(collected_data["all_latents"], dim=1) 
            log_probs = torch.stack(collected_data["all_log_probs"], dim=1)  
            timesteps = torch.stack(collected_data["all_timesteps"]).unsqueeze(0).repeat(config.sample.train_batch_size, 1)
            images = collected_data["images"]
            image_latents = collected_data["image_latents"]
            images2 = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images2 = images2.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images2 = [Image.fromarray(image) for image in images2]

            for idx, img in enumerate(images2):
                img.save(f'images/qwen_{idx}_{rank}.png')

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, ref_images=ref_images, only_strict=False)
            # yield to to make sure reward computation starts
            time.sleep(0)
            

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": collected_data["prompt_embeds"],
                    "prompt_embeds_mask": collected_data["prompt_embeds_mask"],
                    "negative_prompt_embeds": collected_data["negative_prompt_embeds"],
                    "negative_prompt_embeds_mask": collected_data["negative_prompt_embeds_mask"],
                    "timesteps": timesteps,
                    "image_latents": image_latents,
                    "calculated_shape": collected_data["calculated_shape"],
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )
        max_prompt_embeds_len = max([sample["prompt_embeds_mask"].shape[1] for sample in samples])
        max_neg_prompt_embeds_len = max([sample["negative_prompt_embeds_mask"].shape[1] for sample in samples])
        
        if rank == 0:
            print("Sampling Time Total:",time.time()-start_time)
            start_time = time.time()
        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=local_rank!=0,
            position=0,
        ):
            # pad prompt embeds and mask
            seq_pad_len = max_prompt_embeds_len - sample["prompt_embeds"].shape[1]
            sample["prompt_embeds"] = torch.nn.functional.pad(
                sample["prompt_embeds"],  # [B, L, D]
                (0, 0, 0, seq_pad_len),   # pad dim=1 (L)
                value=0,
            )
            sample["prompt_embeds_mask"] = torch.nn.functional.pad(
                sample["prompt_embeds_mask"],  # [B, L]
                (0, seq_pad_len),              # pad dim=1 (L)
                value=0,
            )
            seq_pad_len = max_neg_prompt_embeds_len - sample["negative_prompt_embeds"].shape[1]
            sample["negative_prompt_embeds"] = torch.nn.functional.pad(
                sample["negative_prompt_embeds"],  # [B, L, D]
                (0, 0, 0, seq_pad_len),            # pad dim=1 (L)
                value=0,
            )
            sample["negative_prompt_embeds_mask"] = torch.nn.functional.pad(
                sample["negative_prompt_embeds_mask"],  # [B, L]
                (0, seq_pad_len),                       # pad dim=1 (L)
                value=0,
            )

            rewards, reward_metadata = sample["rewards"].result()
            # print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=device).float()
                for key, value in rewards.items()
            }

        for k in samples[0].keys():
            if not isinstance(samples[0][k], dict):
                try:
                    data = torch.cat([s[k] for s in samples])
                except:
                    print(k)
                    print([s[k].shape for s in samples])

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and rank == 0:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                    rwd = rewards['avg'][i]
                    pil.save(f'images/{idx}_{rwd}.png')

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # gather rewards across processes
        gathered_rewards = {key: gather_tensor(value, world_size) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().float().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if rank == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = gather_tensor(samples["prompt_ids"], world_size).cpu().float().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if local_rank == 0:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if rank == 0:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(world_size, -1, advantages.shape[-1])[rank]
            .to(device)
        )
        if local_rank == 0:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        gradient_accumulation_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        if rank == 0:
            print("Reward Calculation:", time.time()-start_time)
            start_time = time.time()

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=local_rank != 0,
            ):
                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=local_rank != 0,
                ):
                    # Manual gradient accumulation for FSDP
                    if (i * num_train_timesteps + j + 1) % gradient_accumulation_steps == 0:
                        should_sync = True
                    else:
                        should_sync = False
                    
                    with autocast():
                        prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, config, rank)
                        if config.train.beta > 0:
                            with torch.no_grad():
                                if config.use_lora:
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, config, rank)
                                else:
                                    _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer_ref, pipeline, sample, j, config, rank)
                    # grpo logic
                    # print('**** Before clip, Advantages:', sample["advantages"][:, j])
                    advantages = torch.clamp(
                        sample["advantages"][:, j],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                    print("ratio", ratio)
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - config.train.clip_range,
                        1.0 + config.train.clip_range,
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    policy_loss = policy_loss / gradient_accumulation_steps
                    if config.train.beta > 0:
                        kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t ** 2)
                        kl_loss = torch.mean(kl_loss)
                        kl_loss = kl_loss / gradient_accumulation_steps
                        loss = policy_loss + config.train.beta * kl_loss
                    else:
                        loss = policy_loss

                    info["approx_kl"].append(
                        0.5
                        * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                    )
                    info["clipfrac"].append(
                        torch.mean(
                            (
                                torch.abs(ratio - 1.0) > config.train.clip_range
                            ).float()
                        )
                    )
                    info["clipfrac_gt_one"].append(
                        torch.mean(
                            (
                                ratio - 1.0 > config.train.clip_range
                            ).float()
                        )
                    )
                    info["clipfrac_lt_one"].append(
                        torch.mean(
                            (
                                1.0 - ratio > config.train.clip_range
                            ).float()
                        )
                    )
                    info["policy_loss"].append(policy_loss)
                    if config.train.beta > 0:
                        info["kl_loss"].append(kl_loss)

                    info["loss"].append(loss)

                    # backward pass
                    loss.backward()
                    if should_sync:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if should_sync:
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        # Reduce info across processes
                        if is_distributed:
                            for k, v in info.items():
                                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                                info[k] = v / world_size
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if rank == 0:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert should_sync
        if rank == 0:
            print("Training Time:", time.time()-start_time)
            start_time = time.time()
        
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)

