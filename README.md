# ML Training and Inference Suite

This repository contains a suite of tools and scripts for training and inference of machine learning models.
The suite includes functionalities for data preprocessing, model training, evaluation, and deployment.

## Features
- Modular design for easy integration and extension
- torch.compile (with memory constraints)
- Automatic Mixed Precision (AMP) support
- Distributed training capabilities
- WandB integration for experiment tracking
- Configurable via YAML files
- Checkpointing and resuming training support
- Time keeping for graceful shutdowns and resuming on HPCs

## Instructions
1. Clone the repository
2. Install the required dependencies in a conda environment

```bash
conda create -n train_env python=3.13
conda activate train_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

3. Create a .env file with necessary environment variables

```bash
WANDB_API_KEY=your_api_key_here
BASE_DIR=/path/to/your/base/dir
DATA_DIR=/path/to/your/data/dir
```

4. (Optional) Set up WandB account and project for experiment tracking
5. Add your model code in the models directory and update the ``get_model`` function in `vphysics/models/model_utils.py`
6. Create your dataset code. Make sure the dataset returns a tuple (input, target) for each item or change the code in train and eval.

7. Create a dir for your results
8. Copy the config file "vphysics/train/train.yml" to your results dir and modify it as needed:
- Make sure to set the correct model params
- use the correct wandb project name and entity

9. Run the vphysics/train/scripts/train_riv.sh with your changes

## Notes

- The training script uses number of gradient updates instead of epochs to determine training duration. This is to ensure consistent training time across different dataset sizes and batch sizes. The config parameter `updates_per_epoch` controls how many updates are done per evaluation cycle.
- The script supports resuming training from a checkpoint. If a checkpoint is found in the results directory, training will resume from the latest checkpoint. You can also specify a specific checkpoint to resume from in the config file.
- Automatic Mixed Precision (AMP) is enabled by default for faster training. You can disable it in the config file if needed. Not all models support AMP. Same thing for torch.compile. Use the ``mem_budget`` parameter to control memory usage. With this, you can run a model with a larger batch size than your GPU memory would normally allow at the cost of some additional computation time.
- The suite includes chained LR schedulers (up to three). You can configure them in the config file. The first scheduler is applied first, followed by the second, and then the third. They also work on restarting from a checkpoint.


## Suggestions for starters

- Try AMP and torch.compile, they can give significant speedups with minimal code changes
- Use WandB for tracking experiments, it provides a lot of useful features for monitoring and comparing experiments
- Use the checkpointing feature to save and resume training, especially for long-running jobs
- Linear warmup followed by cosine annealing is a good default choice for LR scheduling
- For unstable training, try gradient clipping in the config file