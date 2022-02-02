from torch.nn import Flatten, Sequential

from .vgg_blocks import vgg_conv_block, vgg_final_block


def get_vgg_16(batch_norm: bool = False) -> Sequential:
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
        Flatten(),
        # Fifth block: conv3-512, conv3-512, conv3-512, maxpool
        *vgg_conv_block(
            init_in_channels=1,
            out_channels=512,
            conv_reps=3,
            batch_norm=batch_norm,
        ),
        # FC layers: 4096, 4096, 1000
        *vgg_final_block(),
    )
