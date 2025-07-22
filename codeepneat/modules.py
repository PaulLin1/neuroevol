import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, **kwargs):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # default to "same" padding
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=None, **kwargs):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out += identity
        return self.relu(out)


class PoolConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_type='max', pool_kernel=2, kernel_size=3,
                 stride=1, padding=None, **kwargs):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel)
        else:
            raise ValueError(f"Unsupported pool type: {pool_type}")

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
