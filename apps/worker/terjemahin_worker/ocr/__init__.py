from ..registry import OCRS, load_plugins
from .base import OCR, OcrResult

load_plugins(__name__, __path__)

__all__ = ["OCRS", "OCR", "OcrResult"]
