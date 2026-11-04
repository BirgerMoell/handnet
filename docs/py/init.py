"""handnet: neural networks whose weights are written by hand, not trained.

Each module is one network. Import the one you want:

    from handnet.text_quality import HandNet      # document quality 0-100, any script
    from handnet import sentiment                 # sentiment.score("not a bad film"), sentiment.predict(...)
    from handnet import mnist                     # mnist.predict(images), mnist.explain(image)
    from handnet.mnist_holes import MNISTNet     # MNIST, second design: hole finder + sketches + evidence (83.3%)
"""
from .text_quality import HandNet as TextQualityNet  # noqa: F401
from .mnist_holes import MNISTNet as MNISTHoleNet  # noqa: F401

__all__ = ["TextQualityNet", "MNISTHoleNet"]
