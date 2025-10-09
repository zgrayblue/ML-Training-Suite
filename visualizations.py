# temp for the temp because can't get notebook to work with vscode

# imports 
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# load in data set
data_dir = /scratch/pdy2bw/ml_suite_data
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1] floats
])
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(next(iter(test_loader))[0].shape)

# load in model

# load weights into model

# visualize