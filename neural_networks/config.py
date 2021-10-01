from typing import Union

from pydantic import BaseSettings


class ProjectBaseSettings(BaseSettings):
    DATA_DIR: str = "./data/"

    GPU: str = ""  # TODO - should this be dynamic?


class FullyConnectedSettings(ProjectBaseSettings):
    BATCH_SIZE: int = 64
    DATA_USE_SUBSET: bool = True


# For type checking purposes:
Settings = Union[FullyConnectedSettings]
