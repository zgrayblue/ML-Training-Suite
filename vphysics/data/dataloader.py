"""DataLoader for the WellDatasets.

By: Florian Wiesner
Date: 2025-09-11
"""

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
)
import torch

from the_well.data.datasets import WellDataset


def get_dataloader(
    dataset: WellDataset,
    seed: int,
    batch_size: int,
    num_workers: int,
    is_distributed: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """Get a dataloader for the dataset.

    Parameters
    ----------
    dataset : WellDataset
        Dataset to load.
    seed : int
        Seed for the dataset.
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers.
    is_distributed : bool
        Whether to use distributed sampling
    shuffle : bool
        Whether to shuffle the dataset
    """

    if is_distributed:
        sampler = DistributedSampler(dataset, seed=seed, shuffle=shuffle)
    else:
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = RandomSampler(dataset, generator=generator)
        else:
            sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )

    return dataloader
