import pytorch_lightning as pl
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer

from .resnet_models import (
    get_resnet18,
    get_resnet34,
    get_resnet50,
    get_resnet101,
    get_resnet152,
)


class ResNet(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.network = get_resnet18()
        self.criterion = CrossEntropyLoss()

    def forward(self, x) -> Tensor:
        x = self.flatten(x)
        return self.network(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        # training_step() is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)  # Unrow training examples

        y_hat = self.network(x)
        loss = self.criterion(y_hat, y)

        # Logging to TensorBoard
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return SGD(self.parameters(), lr=1e-3)


class ResNet18:
    def __init__(self):
        super().__init__()


class ResNet34:
    def __init__(self):
        super().__init__()

        self.network = get_resnet34()


class ResNet50:
    def __init__(self):
        super().__init__()

        self.network = get_resnet50()


class ResNet101:
    def __init__(self):
        super().__init__()

        self.network = get_resnet101()


class ResNet152:
    def __init__(self):
        super().__init__()

        self.network = get_resnet152()
