"""handnet: neural networks whose weights are written by hand, not trained.

Each module is one network. Import the one you want:

    from handnet.text_quality import HandNet     # text quality 0-100, any language
    from handnet.mnist import MNISTNet           # MNIST digit classifier
"""
from .text_quality import HandNet as TextQualityNet  # noqa: F401
from .mnist import MNISTNet  # noqa: F401

__all__ = ["TextQualityNet", "MNISTNet"]
