import torch
from ml_suite.models.one_layer import SimpleOneLayerModel

def get_model(model_config: dict) -> torch.nn.Module:
    input_dim = model_config['input_dim']
    output_dim = model_config['output_dim']
    # initialize model with the params and return the model
    return SimpleOneLayerModel(input_dim=input_dim, output_dim=output_dim)  # Placeholder for actual model creation logic

## look up how to make "extensible" so don't have to manually change with model switch, previous chat in chatGPT