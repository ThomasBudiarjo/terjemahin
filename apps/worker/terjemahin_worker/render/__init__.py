from ..registry import RENDERERS, load_plugins
from .base import Renderer

load_plugins(__name__, __path__)

__all__ = ["RENDERERS", "Renderer"]
