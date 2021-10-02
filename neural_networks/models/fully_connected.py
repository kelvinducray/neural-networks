from torch.nn import Linear, ReLU, Sequential


def init_fully_connected() -> Sequential:
    return Sequential(
        Linear(28 * 28, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 10),
    )
