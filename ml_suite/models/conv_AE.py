import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
        super().__init__()
        modules = []
        in_channels = layers[0]
        for out_channels in layers[1:]:
            modules.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )  # Keep padding=1 for same-sized convolutions
            modules.append(act_fn)
            in_channels = out_channels
        modules.append(
            nn.Conv2d(layers[-1], latent_dim, kernel_size=3, stride = 2, padding=1)
        )  # Bottleneck layer
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


# class Decoder(nn.Module):    # no deconv
#     def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
#         super().__init__()

#         self.in_channels = layers[-1]
#         self.latent_dim = latent_dim

#         modules = []
#         in_channels = latent_dim #layers[-1]

#         # Initial convolution layer for latent vector
#         # modules.append(nn.Conv2d(latent_dim, in_channels, kernel_size=3, padding=1))

#         # Iteratively create resize-convolution layers
#         for out_channels in reversed(layers): #layers[:-1]
#             modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))  # Resizing
#             modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  # Convolution
#             modules.append(act_fn)  # Activation function
#             in_channels = out_channels
            
#         # modules.pop() # final activation linear
#         # modules.append(nn.Sigmoid())
        
#         self.conv = nn.Sequential(*modules)

#     def forward(self, x):
#         return self.conv(x)

class Decoder(nn.Module):    # no deconv
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU(), final_act_fn=None):
        super().__init__()

        self.in_channels = layers[-1]
        self.latent_dim = latent_dim

        modules = []
        in_channels = latent_dim

        for out_channels in reversed(layers):
            modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            modules.append(act_fn)
            in_channels = out_channels

        if final_act_fn is not None:
            modules.append(final_act_fn)

        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, layers=[3, 8], latent_dim=8, act_fn=nn.ReLU(), final_act_fn=nn.Sigmoid()):
        super().__init__()
        self.encoder = Encoder(layers=layers, latent_dim=latent_dim, act_fn=act_fn)
        self.decoder = Decoder(layers=layers, latent_dim=latent_dim, act_fn=act_fn, final_act_fn=final_act_fn)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
