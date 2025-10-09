import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# create dataset class, ask chat gpt

from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CIFARDataset(Dataset):
    def __init__(self, config: dict, split: str = "train"):
        dataset_config = config  # already the dataset section

        # Compose transforms
        transform_list = [transforms.ToTensor()]
        if dataset_config.get("normalize", False):
            transform_list.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            )
        self.transform = transforms.Compose(transform_list)

        # Determine train/test split
        is_train = split == "train"

        # Get data directory
        data_dir = dataset_config["data_dir"]

        # Load CIFAR-10 dataset
        self.dataset = datasets.CIFAR10(
            root=data_dir,
            train=is_train,
            download=True,
            transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # ignore label
        return img, img  # input, target


def get_dataset(config: dict, split: str = "train") -> torch.utils.data.Dataset:
    dataset_config = config  # already the 'dataset' section
    train_split = dataset_config.get("train_split", None)

    if split in ["train", "val"] and train_split is not None:
        full_dataset = CIFARDataset(dataset_config, split="train")
        train_ratio = float(train_split)
        train_len = int(len(full_dataset) * train_ratio)
        val_len = len(full_dataset) - train_len
        train_set, val_set = torch.utils.data.random_split(
            full_dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(dataset_config.get("seed", 42))
        )
        return train_set if split == "train" else val_set
    else:
        return CIFARDataset(dataset_config, split)




# def get_dataset(config: dict, split: str="train") -> torch.utils.data.Dataset:
#     # Make sure to split your data somehow, either return two datasets here or use the split option
#     return torch.utils.data.Dataset()  # Placeholder for actual dataset implementation, (return class here)

