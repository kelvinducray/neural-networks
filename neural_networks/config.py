from typing import Union

from pydantic import BaseSettings


class FullyConnectedSettings(BaseSettings):
    BATCH_SIZE: int = 64

    DATA_USE_SUBSET: bool = True
    DATA_DIR: str = "./data/"


# For type checking purposes:
Settings = Union[FullyConnectedSettings]
