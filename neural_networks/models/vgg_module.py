import pytorch_lightning as pl
from torch import Tensor
from torch.nn import CrossEntropyLoss, Flatten
from torch.optim import SGD, Optimizer

from .vgg_models import get_vgg16, get_vgg19


class VGG(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.network = get_vgg16()
        self.criterion = CrossEntropyLoss()
        self.flatten = Flatten()

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


class VGG16(VGG):
    def __init__(self):
        super().__init__()


class VGG19(VGG):
    def __init__(self):
        super().__init__()

        self.network = get_vgg19()
