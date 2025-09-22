import torch
import torch.nn as nn

class SimpleOneLayerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleOneLayerModel, self).__init__()
        # one linear layer
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # x is expected to have shape [batch_size, input_dim]
        out = self.linear(x)
        return out
