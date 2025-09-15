"""Optimizer utilities.

By: Florian Wiesner
Date: 2025-09-11
"""

import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create an optimizer.

    Parameters
    ----------
    model : nn.Module
        The model to optimize
    config : dict
        Configuration dictionary for the optimizer

    Returns
    -------
    torch.optim.Optimizer
        Optimizer
    """
    lr = float(config["learning_rate"])
    name = config["name"]

    if name == "AdamW":
        weight_decay = config.get("weight_decay", 0)
        betas = config.get("betas", (0.9, 0.999))

        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif name == "Adam":
        betas = config.get("betas", (0.9, 0.999))

        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
        )
    else:
        raise ValueError(f"Optimizer {name} not supported")

    return optimizer
