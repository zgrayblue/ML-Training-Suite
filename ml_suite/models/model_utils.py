import torch
from ml_suite.models.unet import SimpleOneLayerModel

def get_model(model_config: dict) -> torch.nn.Module:
    input_dim = model_config['input_dim']
    output_dim = model_config['output_dim']
    # initialize model with the params and return the model
    return torch.nn.Module()  # Placeholder for actual model creation logic
