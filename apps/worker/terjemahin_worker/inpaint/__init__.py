from ..registry import INPAINTERS, load_plugins
from .base import Inpainter

load_plugins(__name__, __path__)

__all__ = ["INPAINTERS", "Inpainter"]
