import torch
import torch.nn as nn
import torch.nn.functional as F

from pixelflow.layers import LambdaLayer, ElementwiseParams2d
from pixelflow.layers.autoregressive import MaskedConv2d


class DropMaskedResidualBlock2d(nn.Module):

    def __init__(self, h, dropout=0.0, kernel_size=3, data_channels=3):
        super(DropMaskedResidualBlock2d, self).__init__()

        self.conv1 = MaskedConv2d(2 * h, h, kernel_size=1, mask_type='B', data_channels=data_channels)
        self.conv2 = MaskedConv2d(h, h, kernel_size=kernel_size, padding=kernel_size//2, mask_type='B', data_channels=data_channels)
        self.conv3 = MaskedConv2d(h, 2 * h, kernel_size=1, mask_type='B', data_channels=data_channels)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        identity = x

        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))

        return self.drop(x) + identity


class DropPixelCNN(nn.Sequential):

    def __init__(self, in_channels, num_params, filters=128, num_blocks=15, output_filters=1024, kernel_size=3, kernel_size_in=7, dropout=0.0, init_transforms=lambda x: x):

        layers = [LambdaLayer(init_transforms)] +\
                 [MaskedConv2d(in_channels, 2 * filters, kernel_size=kernel_size_in, padding=kernel_size_in//2, mask_type='A', data_channels=in_channels)] +\
                 [DropMaskedResidualBlock2d(filters, dropout=dropout, data_channels=in_channels, kernel_size=kernel_size) for _ in range(num_blocks)] +\
                 [nn.ReLU(True), MaskedConv2d(2 * filters, output_filters, kernel_size=1, mask_type='B', data_channels=in_channels)] +\
                 [nn.ReLU(True), MaskedConv2d(output_filters, num_params * in_channels, kernel_size=1, mask_type='B', data_channels=in_channels)] +\
                 [ElementwiseParams2d(num_params)]

        super(DropPixelCNN, self).__init__(*layers)
