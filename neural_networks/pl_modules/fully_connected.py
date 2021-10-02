import pytorch_lightning as pl
from torch import Tensor
from torch.nn import CrossEntropyLoss, Flatten
from torch.optim import SGD, Optimizer

from ..models.fully_connected import init_fully_connected


class FullyConnectedModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = Flatten()
        self.network = init_fully_connected()
        self.criterion = CrossEntropyLoss()
        self.type_check = False

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
