from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)


def vgg_conv_block(
    init_in_channels: int,
    out_channels: int,
    conv_reps: int,
    batch_norm: bool,
) -> list[Module]:
    layers = []
    in_channels = init_in_channels

    # Iteratively add conv. layers
    for _ in range(conv_reps):
        layers.extend(
            [
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                ReLU(),
            ]
        )
        if batch_norm:
            layers.append(BatchNorm2d())
        in_channels = out_channels
    # Finally add max pooling layer
    layers.append(MaxPool2d(kernel_size=2, stride=2))

    return layers


def vgg_final_block() -> list[Module]:
    return [
        Linear(512 * 7 * 7, 4096),
        ReLU(),
        Dropout(0.5),  # Hyperparam - add to settings?
        Linear(4096, 4096),
        ReLU(),
        Dropout(0.5),
        Linear(4096, 10),  # TODO: Change to number of classes
    ]


def vgg_16(batch_norm: bool = False):
    """
    Coded from the original paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR
    LARGE-SCALE IMAGE RECOGNITION" by Simonyan & Zisserman (2015)
    Link to full PDF: https://arxiv.org/pdf/1409.1556.pdf
    """

    # NOTE: Start by just coding a VGG16

    return Sequential(
        # Input: 224x224 RGB Image
        # First block: conv3-64, conv3-64, maxpool
        *vgg_conv_block(
            init_in_channels=3,
            out_channels=64,
            conv_reps=2,
            batch_norm=batch_norm,
        ),
        # Second block: conv3-128, conv3-128, maxpool
        *vgg_conv_block(
            init_in_channels=1,
            out_channels=128,
            conv_reps=2,
            batch_norm=batch_norm,
        ),
        # Third block: conv3-256, conv3-256, conv3-256, maxpool
        *vgg_conv_block(
            init_in_channels=1,
            out_channels=256,
            conv_reps=3,
            batch_norm=batch_norm,
        ),
        # Fourth block: conv3-512, conv3-512, conv3-512, maxpool
        *vgg_conv_block(
            init_in_channels=1,
            out_channels=512,
            conv_reps=3,
            batch_norm=batch_norm,
        ),
        # Flatten before fully connected layers
        Flatten()
        # Fifth block: conv3-512, conv3-512, conv3-512, maxpool
        * vgg_conv_block(
            init_in_channels=1,
            out_channels=512,
            conv_reps=3,
            batch_norm=batch_norm,
        ),
        # FC layers: 4096, 4096, 1000
        *vgg_final_block,
    )
