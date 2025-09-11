import argparse
import platform
import os
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._functorch.config as functorch_config

import yaml
from yaml import CLoader

from vphysics.models.model_utils import get_model
from vphysics.train.train_base import Trainer
from vphysics.train.utils.optimizer import get_optimizer
from vphysics.train.utils.lr_scheduler import get_lr_scheduler
from vphysics.data.dataloader import get_dataloader
from vphysics.train.utils.checkpoint_utils import load_checkpoint
from vphysics.train.utils.wandb_logger import WandbLogger


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=CLoader)
    return config


def time_str_to_seconds(time_str: str) -> float:
    return sum(x * int(t) for x, t in zip([3600, 60, 1], time_str.split(":")))


def get_checkpoint_path(output_dir: Path, checkpoint_name: str) -> Path:
    if checkpoint_name == "latest":
        checkpoint_path = output_dir / "latest.pt"
    elif checkpoint_name == "best":
        checkpoint_path = output_dir / "best.pt"
    elif checkpoint_name.isdigit():
        checkpoint_path = output_dir / f"epoch_{checkpoint_name}/checkpoint.pt"
    else:
        raise ValueError(f"Invalid checkpoint name: {checkpoint_name}")
    return checkpoint_path


@record
def main(
    config_path: Path,
):
    load_dotenv()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    config = load_config(config_path)
    output_dir = config_path.parent

    time_limit = config.get("time_limit", None)
    if time_limit is not None:
        time_limit = time_str_to_seconds(time_limit)

    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ############################################################
    ###### Load torch modules ##################################
    ############################################################

    model = get_model(config["model"])
    optimizer = get_optimizer(model, config["optimizer"])
    lr_scheduler = get_lr_scheduler(optimizer, config["lr_scheduler"])

    ############################################################
    ###### Load datasets and dataloaders #######################
    ############################################################

    train_dataloader = get_dataloader(
        dataset=dataset,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        is_distributed=dist.is_initialized(),
        shuffle=True,
    )
    val_dataloader = get_dataloader(
        dataset=dataset,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        is_distributed=dist.is_initialized(),
        shuffle=False,
    )
    total_updates = int(config["total_updates"])
    updates_per_epoch = int(config["updates_per_epoch"])
    cp_every_updates = int(config["checkpoint_every_updates"])
    wandb_logger = WandbLogger(config["wandb"])

    ############################################################
    ###### Load checkpoint #####################################
    ############################################################

    checkpoint_name = config.get("checkpoint_name", None)
    if checkpoint_name is not None:
        checkpoint_path = get_checkpoint_path(output_dir, checkpoint_name)
        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        grad_scaler_sd = checkpoint["grad_scaler_state_dict"]

        samples_trained = checkpoint["samples_trained"]
        batches_trained = checkpoint["batches_trained"]
        epoch = checkpoint["epoch"]

    ############################################################
    ###### Compile and distribute model #########################
    ############################################################
    model.to(device)
    functorch_config.activation_memory_budget = config.get("mem_budget", 1)
    if not platform.system() == "Windows":
        model = torch.compile(model, mode="max-autotune")
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=device,
        )

    wandb_logger.watch(model, criterion=torch.nn.MSELoss())

    ############################################################
    ###### Initialize trainer ##################################
    ############################################################

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        total_updates=total_updates,
        updates_per_epoch=updates_per_epoch,
        checkpoint_every_updates=cp_every_updates,
        output_dir=output_dir,
        wandb_logger=wandb_logger,
        time_limit=time_limit,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )

    if checkpoint_name is not None:
        trainer.state.samples_trained = samples_trained
        trainer.state.batches_trained = batches_trained
        trainer.state.epoch = epoch
        if grad_scaler_sd is not None:
            trainer.scaler.load_state_dict(grad_scaler_sd)

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config_path)

    main(config_path)
