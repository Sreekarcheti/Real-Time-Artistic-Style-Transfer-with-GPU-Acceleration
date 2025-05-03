
import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, 3, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(channels, channels, 3, 1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),

            ConvLayer(32, 64, 3, 2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            ConvLayer(64, 128, 3, 2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),

            UpsampleConvLayer(128, 64, 3, 1, upsample=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            UpsampleConvLayer(64, 32, 3, 1, upsample=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),

            ConvLayer(32, 3, 9, 1)
        )

    def forward(self, x):
        return self.model(x)

