"""
backend/models/base.py
───────────────────────
Abstract interface that every binarization model must implement.

To plug in your own model:
  1. Create a new file in backend/models/
  2. Subclass BinarizationModel
  3. Implement predict()
  4. Register it in backend/pipeline.py
"""

from abc import ABC, abstractmethod
from PIL import Image


class BinarizationModel(ABC):
    """
    Contract for all binarization backends.

    Inputs:  RGB / grayscale PIL Image (any size)
    Outputs: Grayscale PIL Image ("L" mode), same size as input
    """

    @abstractmethod
    def predict(self, img: Image.Image) -> Image.Image:
        """
        Run binarization.

        Args:
            img: Input document image (PIL, any mode/size)

        Returns:
            Binarized PIL Image in "L" mode, same pixel dimensions as input.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable model name for logging."""
        return self.__class__.__name__
