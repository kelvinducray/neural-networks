import torch
from torch.nn import BatchNorm2d, Conv2d, Module, ReLU


class block(Module):
    def __init__(self):
        pass


def resnet_block(
    in_channels,
    out_channels,
    stride,
    identity_downsample=None,
) -> list[Module]:
    modules = [
        Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
        BatchNorm2d(out_channels),
        Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        ),
        BatchNorm2d(out_channels),
        Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1),
        BatchNorm2d(out_channels * 4),
        ReLU(),
    ]

    if identity_downsample is not None:
        pass

    return modules
