import torch
import torch.nn as nn
from ml_suite.models.conv_AE import ConvolutionalAutoencoder

# Map from string to nn activation class
ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
    "none": nn.Identity
}

def get_activation(name: str):
    name = name.lower()
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_MAP[name]()

def get_model(model_config: dict) -> nn.Module:
    name = model_config.get("name", "").lower()

    if name == "conv_ae":
        layers = model_config["layers"]
        latent_dim = model_config["latent_dim"]
        act_fn = get_activation(model_config.get("act_fn", "relu"))
        final_act_fn = get_activation(model_config.get("final_act_fn", "none"))

        model = ConvolutionalAutoencoder(
            layers=layers,
            latent_dim=latent_dim,
            act_fn=act_fn,
            final_act_fn=final_act_fn
        )

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model