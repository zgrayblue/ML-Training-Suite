# verifiable-physics
Improve long-horizon physics predictions with verifyable rewards RL

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install the_well wandb python-dotenv pytest
```

Create a `.env` file with your Weights & Biases API key:

```
WANDB_API_KEY=your_api_key_here
```

## Download Datasets

```bash
the-well-download --base-path ./data --dataset turbulent_radiative_layer_2D --parallel
```