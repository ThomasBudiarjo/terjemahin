"""Plugin registry: every pipeline stage has several interchangeable methods.

A method is registered with a factory `(settings, **opts) -> instance`. Heavy dependencies
(torch, transformers, API SDKs) must be imported inside the factory or the instance, never at
module import time, so listing methods stays cheap and a missing optional dependency only
disables the methods that need it.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

STAGES = ("detector", "ocr", "translator", "mask_refiner", "inpainter", "renderer")


@dataclass
class MethodInfo:
    name: str
    stage: str
    factory: Callable[..., Any]
    description: str = ""
    requires: tuple[str, ...] = ()  # importable python modules
    weights: tuple[str, ...] = ()  # keys in weights.WEIGHTS (auto-downloadable)
    needs_key: str | None = None  # Settings attribute that must be non-empty (API key)
    offline: bool = True  # runs without network once weights are present

    def status(self, settings: Settings) -> tuple[bool, str]:
        """(available, reason). Weights are not required to exist: they download on demand."""
        for mod in self.requires:
            if importlib.util.find_spec(mod) is None:
                return False, f"python package '{mod}' not installed"
        if self.needs_key and not getattr(settings, self.needs_key, ""):
            return False, f"{self.needs_key.upper()} not set"
        return True, "ok"


@dataclass
class Registry:
    stage: str
    methods: dict[str, MethodInfo] = field(default_factory=dict)

    def register(
        self,
        name: str,
        *,
        description: str = "",
        requires: tuple[str, ...] = (),
        weights: tuple[str, ...] = (),
        needs_key: str | None = None,
        offline: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
            if name in self.methods:
                raise ValueError(f"{self.stage} method '{name}' registered twice")
            self.methods[name] = MethodInfo(
                name, self.stage, factory, description, requires, weights, needs_key, offline
            )
            return factory

        return deco

    def get(self, name: str) -> MethodInfo:
        try:
            return self.methods[name]
        except KeyError:
            raise ValueError(
                f"unknown {self.stage} '{name}', choose from: {', '.join(sorted(self.methods))}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self.methods)

    def build(self, name: str, settings: Settings, **opts: Any) -> Any:
        info = self.get(name)
        ok, reason = info.status(settings)
        if not ok:
            raise RuntimeError(f"{self.stage} '{name}' unavailable: {reason}")
        return info.factory(settings, **opts)


DETECTORS = Registry("detector")
OCRS = Registry("ocr")
TRANSLATORS = Registry("translator")
MASK_REFINERS = Registry("mask_refiner")
INPAINTERS = Registry("inpainter")
RENDERERS = Registry("renderer")

REGISTRIES: dict[str, Registry] = {
    "detector": DETECTORS,
    "ocr": OCRS,
    "translator": TRANSLATORS,
    "mask_refiner": MASK_REFINERS,
    "inpainter": INPAINTERS,
    "renderer": RENDERERS,
}


def load_plugins(package: str, path: list[str]) -> None:
    """Import every submodule of a stage package so their @register decorators run."""
    for m in pkgutil.iter_modules(path):
        if m.name.startswith("_") or m.name == "base":
            continue
        try:
            importlib.import_module(f"{package}.{m.name}")
        except Exception:  # a broken plugin must not take the others down
            log.exception("failed to load plugin %s.%s", package, m.name)


def load_all() -> dict[str, Registry]:
    from . import detect, inpaint, mask, ocr, render, translate  # noqa: F401

    return REGISTRIES
