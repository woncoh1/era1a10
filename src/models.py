import torch
import torch.nn as nn


DROPOUT = 0.05


def prepblock( # Preparation block
    i:int, # in_channels
    o:int, # out_channels
) -> nn.Sequential:
    "Use 3x3 convolution to create more channels."
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, padding_mode='replicate', bias=False),
        nn.BatchNorm2d(o),
        nn.Dropout(p=DROPOUT),
        nn.ReLU(),
    )


def convblock( # Convolution block
    i:int, # in_channels
    o:int, # out_channels
) -> nn.Sequential:
    "Use 3x3 convolution to extract features."
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, padding_mode='replicate', bias=False),
        nn.MaxPool2d(2, stride=2),
        nn.BatchNorm2d(o),
        nn.Dropout(p=DROPOUT),
        nn.ReLU(),
    )


def resblock( # Residual block
    i:int, # in_channels
    o:int, # out_channels
) -> nn.Sequential:
    "Use two 3x3 convolution layers for the residual path."
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, padding_mode='replicate', bias=False),
        nn.BatchNorm2d(o),
        nn.Dropout(p=DROPOUT),
        nn.ReLU(),
        nn.Conv2d(o, o, 3, padding=1, padding_mode='replicate', bias=False),
        nn.BatchNorm2d(o),
        nn.Dropout(p=DROPOUT),
        nn.ReLU(),
    )


def predblock( # Prediction block
    i:int, # in_channels
    o:int, # out_channels
) -> nn.Sequential:
    "Use max pooling and 1x1 convolution to compute logit for loss function."
    return nn.Sequential(
        # [-1, i, s, s]
        nn.MaxPool2d(4),
        # [-1, i, 1, 1]
        nn.Conv2d(i, o, 1, padding=0, bias=False),
        # [-1, o, 1, 1]
        nn.Flatten(),
        # [-1, o]
    )


class SkipBlock(nn.Module):
    "Convolution layers with identity and residual paths"
    def __init__(self,
        i:int, # in_channels
        o:int, # out_channels
    ) -> None:
        super().__init__()
        self.conv = convblock(i, o)
        self.res = resblock(o, o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        y = self.res(x)
        return x + y


class PageNet(nn.Module):
    """ResNet optimized for training speed
    https://github.com/davidcpage/cifar10-fast
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = prepblock(3, 64)
        self.conv1 = SkipBlock(64, 128)
        self.conv2 = convblock(128, 256)
        self.conv3 = SkipBlock(256, 512)
        self.trans = predblock(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.trans(x)
        return x