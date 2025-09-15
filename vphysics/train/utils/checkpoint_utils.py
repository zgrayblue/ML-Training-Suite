from pathlib import Path

import torch
from torch.amp.grad_scaler import GradScaler


def sanitize_model_dict(model: torch.nn.Module) -> dict:
    """Sanitize a model's state dict for saving.

    This function removes any prefixes added by distributed training or compiling
    from the state dict keys. This is useful to ensure compatibility when loading
    the model in different environments.

    Parameters
    ----------
    model : torch.nn.Module
        The model to sanitize.

    Returns
    -------
    dict
        The sanitized state dict.
    """
    state_dict = model.state_dict()
    sanitized_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if "module." in key:
            new_key = key.replace("module.", "")
        if "_orig_mod." in key:
            new_key = key.replace("_orig_mod.", "")
        sanitized_dict[new_key] = value
    return sanitized_dict


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """Load a checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint
    device : torch.device
        Device to load the checkpoint to

    Returns
    -------
    dict
        Checkpoint
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    return checkpoint


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    samples_trained: int,
    batches_trained: int,
    epoch: int,
    grad_scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """Save a checkpoint."""
    checkpoint = {
        "samples_trained": samples_trained,
        "batches_trained": batches_trained,
        "epoch": epoch,
        "model_state_dict": sanitize_model_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": grad_scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
