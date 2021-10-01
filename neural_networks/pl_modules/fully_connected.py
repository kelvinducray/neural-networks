import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss, Flatten

from ..models.fully_connected import init_fully_connected


class FullyConnectedModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.network = init_fully_connected()
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop - it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)
