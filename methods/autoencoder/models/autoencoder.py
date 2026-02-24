from typing import Callable, List, Optional


class Autoencoder:
    """
    TensorFlow/Keras autoencoder implementation has been deprecated.
    """

    def __init__(
        self,
        data_name: str,
        layers: Optional[List] = None,
        optimizer: str = "rmsprop",
        loss: Optional[Callable] = None,
    ) -> None:
        raise RuntimeError(
            "Autoencoder is disabled because TensorFlow/Keras support was removed."
        )
