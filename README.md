# Talk2Move Training Code

This repository contains training scripts for scene-level image editing models using GRPO (Group Relative Policy Optimization).

## Licenses
This codebase is build upon:
- Flow-GRPO, which is licensed under MIT license; 
- Orient-Anything that is licensed under CC-BY-4.0 license, 
- lang-segment-anything that is licensed underApache-2.0 license; 
- Grounding-DINO that is licensed under Apache-2.0 license

## Key Components

### Configuration Files (`config/`)
- **grpo.py**: Main configuration file containing all GRPO training setups for different models and tasks
- **base.py**: Base configuration with default hyperparameters
- **dpo.py**: Direct Preference Optimization configurations
- **sft.py**: Supervised fine-tuning configurations

### Reward Functions (`flow_grpo/`)
The reward system supports multiple scoring methods:
- **Text-Image Alignment**: CLIP score, PickScore, ImageReward
- **Image Quality**: Aesthetic score, JPEG compressibility
- **Task-Specific**: OCR accuracy, GenEval, object manipulation
- **Edit Quality**: Position accuracy, rotation accuracy, resize accuracy, LPIPS similarity
- **Multi-Modal**: Qwen-VL based verification

### Training Scripts (`scripts/`)
- **Single-node training**: Use `train_*.py` directly with accelerate
- **Multi-node training**: Use scripts in `scripts/multi_node/` for distributed training
- **Demo scripts**: Quick inference examples in `scripts/demo/`
- **Test scripts**: Evaluation and testing utilities in `scripts/test/`

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- 16 GPUs (2 nodes × 8 GPUs per node)
- Required Python packages (install via `pip install -e .`)

## Configuration

Before running training, update the paths in your configuration:

1. Replace `enter_path_here` placeholders in the codebase with your actual paths
2. Update `MASTER_ADDR` in `scripts/multi_node/qwenimagedit/main2.sh` to match your master node IP
3. Ensure all nodes can communicate via the specified master address and port

## Running Training (16 GPUs)

To run training on 16 GPUs across 2 nodes (8 GPUs per node):

### On Node 0 (Master):
```bash
sh scripts/multi_node/qwenimagedit/main2.sh 0
```

### On Node 1 (Worker):
```bash
sh scripts/multi_node/qwenimagedit/main2.sh 1
```

## Training Configuration

The training script uses the following default settings:
- **GPUs per node**: 8
- **Number of nodes**: 2
- **Total GPUs**: 16
- **Master port**: 19001
- **Config**: `config/grpo.py:counting_qwenimage_edit_mini`

To modify these settings, edit `scripts/multi_node/qwenimagedit/main2.sh`.

## Available Configurations

Check `config/grpo.py` for available training configurations:

### Qwen-Image-Edit Configurations
- `counting_qwenimage_edit_mini` - Mini configuration for testing
- `counting_qwenimage_edit_8gpu_2` - 8 GPU setup
- Various task-specific configs for rotation, resize, translation, manipulation

Each configuration specifies:
- Model architecture and checkpoint paths
- Batch sizes and gradient accumulation steps
- Sampling parameters (num_steps, guidance_scale)
- Reward function weights
- Training hyperparameters (learning rate, beta, etc.)

## Monitoring Training

Training logs and checkpoints will be saved according to the `save_dir` specified in your configuration (typically in `logs/` directory).

## Customization

### Adding New Reward Functions
1. Create a new scorer in `flow_grpo/` (e.g., `my_scorer.py`)
2. Implement the scorer class with a `run()` or `__call__()` method
3. Add the reward function to `flow_grpo/rewards.py` in the `score_functions` dict
4. Configure the reward weight in your config file

### Creating New Configurations
1. Open `config/grpo.py`
2. Define a new function (e.g., `def my_custom_config()`)
3. Start from an existing config: `config = pickscore_sd3()` or `config = compressibility()`
4. Modify parameters as needed
5. Use it with: `--config config/grpo.py:my_custom_config`

### Multi-Node Setup
For different node counts, modify the shell scripts:
- `GPUS_PER_NODE`: GPUs available per node (typically 8)
- `NUM_MACHINES`: Total number of nodes
- `MASTER_ADDR`: IP address of the master node (rank 0)
- `MASTER_PORT`: Communication port (default 19001)

## Troubleshooting

- **Connection issues**: Verify that `MASTER_ADDR` is correct and nodes can communicate
- **CUDA out of memory**: Reduce batch size in the config file
- **Path errors**: Ensure all `enter_path_here` placeholders are replaced with valid paths
- **Reward server errors**: Check that reward server IPs (`your-api-server-ip`, `your-reward-server-ip`) are correctly configured
- **Import errors**: Run `pip install -e .` to install the package in development mode
- **NCCL timeout**: Increase timeout or check network connectivity between nodes

## Citation

If you use this code in your research, please cite the relevant papers for the models and methods use
```
@misc{tan2026talk2movereinforcementlearningtextinstructed,
      title={Talk2Move: Reinforcement Learning for Text-Instructed Object-Level Geometric Transformation in Scenes}, 
      author={Jing Tan and Zhaoyang Zhang and Yantao Shen and Jiarui Cai and Shuo Yang and Jiajun Wu and Wei Xia and Zhuowen Tu and Stefano Soatto},
      year={2026},
      eprint={2601.02356},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.02356}, 
}
```

## Contribution
This codebase is built by [Jing Tan](https://sparkstj.github.io/) during her internship at AWS Agentic AI. 

For any question, feel free to contact her via 
```
tj023@ie.cuhk.edu.hk
```