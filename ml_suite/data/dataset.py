import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
from the_well.data.datasets import WellDataset
from einops import rearrange


class PhysicsDataset(WellDataset):
    def __init__(
        self,
        data_dir: Path,
        n_steps_input: int, # control how many steps get in x
        n_steps_output: int, # don't need for recon. just dynamics
        use_normalization: bool,
        min_dt_stride: int,
        max_dt_stride: int,
        full_trajectory_mode: bool,
        max_rollout_steps: int,
    ):
        super().__init__(
            path=str(data_dir),
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            full_trajectory_mode=full_trajectory_mode,
            max_rollout_steps=max_rollout_steps,
        )
 
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        data_dict = super().__getitem__(index)
        x = data_dict["input_fields"]  # (n_steps_input, H, W, C)
        y = data_dict["output_fields"]  # (n_steps_output, H, W, C) (next timesteps)

        # reshape here then standardized for all models
        x = rearrange(x, "T H W C -> T C H W") # T here is num ts for x, only can squeeze if 1
        x = x.squeeze(0)
        
        return x, x # return x, y for dynamics training; for reconstruction want x to be target


def get_dataset(
    data_dir: Path,
    n_steps_input: int = 1,
    n_steps_output: int = 1,
    use_normalization: bool = True,
    dt_stride: int | list[int] = 1,
    full_trajectory_mode: bool = False,
    max_rollout_steps: int = 10000,
) -> PhysicsDataset:
    """
    Get a WellDataset.
 
    Parameters
    ----------
    data_dir : Path
        Path to the dataset directory.
    n_steps_input : int
        Number of input time steps.
    n_steps_output : int
        Number of output time steps.
    use_normalization : bool
        Whether to use normalization.
    dt_stride : int or list of int
        If int, fixed stride between time steps.
        If list of int, [min_stride, max_stride] for random stride sampling.
    full_trajectory_mode : bool
        Whether to use full trajectory mode.
    max_rollout_steps : int
        Maximum number of rollout steps for full trajectory mode.
    """
    if isinstance(dt_stride, list):
        min_dt_stride = dt_stride[0]
        max_dt_stride = dt_stride[1]
    else:
        min_dt_stride = dt_stride
        max_dt_stride = dt_stride
 
    return PhysicsDataset(
        data_dir=data_dir,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        use_normalization=use_normalization,
        min_dt_stride=min_dt_stride,
        max_dt_stride=max_dt_stride,
        full_trajectory_mode=full_trajectory_mode,
        max_rollout_steps=max_rollout_steps,
    )


# class CIFARDataset(Dataset):
#     def __init__(self, config: dict, split: str = "train"):
#         dataset_config = config  # already the dataset section

#         # Compose transforms
#         transform_list = [transforms.ToTensor()]
#         if dataset_config.get("normalize", False):
#             transform_list.append(
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                      std=[0.5, 0.5, 0.5])
#             )
#         self.transform = transforms.Compose(transform_list)

#         # Determine train/test split
#         is_train = split == "train"

#         # Get data directory
#         data_dir = dataset_config["data_dir"]

#         # Load CIFAR-10 dataset
#         self.dataset = datasets.CIFAR10(
#             root=data_dir,
#             train=is_train,
#             download=True,
#             transform=self.transform
#         )

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img, _ = self.dataset[idx]  # ignore label
#         return img, img  # input, target


# def get_dataset(config: dict, split: str = "train") -> torch.utils.data.Dataset:
#     dataset_config = config  # already the 'dataset' section
#     train_split = dataset_config.get("train_split", None)

#     if split in ["train", "val"] and train_split is not None:
#         full_dataset = CIFARDataset(dataset_config, split="train")
#         train_ratio = float(train_split)
#         train_len = int(len(full_dataset) * train_ratio)
#         val_len = len(full_dataset) - train_len
#         train_set, val_set = torch.utils.data.random_split(
#             full_dataset,
#             [train_len, val_len],
#             generator=torch.Generator().manual_seed(dataset_config.get("seed", 42))
#         )
#         return train_set if split == "train" else val_set
#     else:
#         return CIFARDataset(dataset_config, split)




# def get_dataset(config: dict, split: str="train") -> torch.utils.data.Dataset:
#     # Make sure to split your data somehow, either return two datasets here or use the split option
#     return torch.utils.data.Dataset()  # Placeholder for actual dataset implementation, (return class here)

