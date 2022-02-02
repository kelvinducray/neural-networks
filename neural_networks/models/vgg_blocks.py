from torch.nn import BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU


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
