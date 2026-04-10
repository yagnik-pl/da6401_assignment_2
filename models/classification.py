"""Classification components."""

from .vgg11 import ClassificationHead, VGG11


class VGG11Classifier(VGG11):
    """Alias kept for the skeleton API."""

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )


__all__ = ["ClassificationHead", "VGG11Classifier"]
