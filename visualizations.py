# =========================
# Imports
# =========================
import sys
sys.path.append("/sfs/gpfs/tardis/home/pdy2bw/Research/ML-Training-Suite")

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from ml_suite.models.conv_AE import ConvolutionalAutoencoder

# =========================
# Paths and device
# =========================
data_dir = "/scratch/pdy2bw/ml_suite_data"
weights_dir = "results/simple_test3/latest.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load dataset (no batching)
# =========================
transform = transforms.ToTensor()
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

# just take the first 5 images
imgs = torch.stack([test_dataset[i][0] for i in range(5)]).to(device)

# =========================
# Load model and weights
# =========================
layers = [3, 8]
latent_dim = 16
act_fn = nn.ReLU()
final_act_fn = nn.Identity()

model = ConvolutionalAutoencoder(layers=layers, latent_dim=latent_dim, act_fn=act_fn, final_act_fn=final_act_fn).to(device)

checkpoint = torch.load(weights_dir, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# =========================
# Run inference
# =========================
with torch.no_grad():
    outputs = model(imgs)
    mse = ((imgs - outputs) ** 2).mean(dim=[1,2,3])
    print(f"MSE for first 5 images: {mse}")

# =========================
# Visualization
# =========================
def save_images(imgs, prefix="image", titles=None, out_dir="reconstructions"):
    os.makedirs(out_dir, exist_ok=True)
    imgs = imgs.cpu().numpy().transpose((0, 2, 3, 1))  # BCHW -> BHWC
    for i, img in enumerate(imgs):
        file_name = f"{prefix}_{i}.png"
        path = os.path.join(out_dir, file_name)
        plt.imsave(path, img)
        if titles:
            print(f"Saved {titles[i]}: {path}")
        else:
            print(f"Saved: {path}")

# Save originals and reconstructions
save_images(imgs, prefix="original", titles=[f"Original" for _ in range(5)])
save_images(outputs, prefix="reconstruction", titles=[f"Reconstruction" for _ in range(5)])