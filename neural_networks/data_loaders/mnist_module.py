from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from ..config import Settings


class MNISTDataModule(LightningDataModule):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def prepare_data(self):
        # Download both train and test data
        for train_bool in [True, False]:
            MNIST(
                self.settings.DATA_DIR,
                train=train_bool,
                download=True,
                transform=ToTensor(),
            )

    def setup(self, stage: Optional[str] = None):
        # Transform
        transform = Compose([ToTensor()])
        mnist_train = MNIST(
            self.settings.DATA_DIR,
            train=True,
            download=False,
            transform=transform,
        )
        mnist_test = MNIST(
            self.settings.DATA_DIR,
            train=False,
            download=False,
            transform=transform,
        )

        # Train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # Assign to use in DataLoaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.settings.BATCH_SIZE,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.settings.BATCH_SIZE,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.settings.BATCH_SIZE,
        )
