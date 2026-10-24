"""Coercion and validation of per-stage options.

Options reach the factories as JSON values from the web/API, or as strings from the CLI's
`--opt stage.key=value`. Factories check them here at build time so a bad value fails before
any stage runs, with a message that names the option. Without that check the error came from
inside numpy, OpenCV or onnxruntime after the earlier stages had already run.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

log = logging.getLogger(__name__)

TRUE = frozenset(("1", "true", "yes", "on", "y", "t"))
FALSE = frozenset(("0", "false", "no", "off", "n", "f"))


def as_bool(v: Any, name: str = "option") -> bool:
    """True/False, 1/0 and the words true/false, yes/no, on/off (any case). Anything else is an
    error: the string 'False' is truthy in Python, which turned `uppercase=False` on."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in TRUE:
            return True
        if s in FALSE:
            return False
    raise ValueError(f"{name} must be true or false, not {v!r}")


def _number(v: Any, name: str, kind: type, lo: float | None, hi: float | None) -> Any:
    if isinstance(v, bool):
        raise ValueError(f"{name} must be a number, not {v!r}")  # noqa: TRY004
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, not {v!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number, not {v!r}")
    if kind is int:
        if x != int(x):
            raise ValueError(f"{name} must be a whole number, not {v!r}")
        x = int(x)
    if lo is not None and x < lo:
        raise ValueError(f"{name} must be >= {lo}, not {v!r}")
    if hi is not None and x > hi:
        raise ValueError(f"{name} must be <= {hi}, not {v!r}")
    return x


def as_int(v: Any, name: str = "option", lo: float | None = None, hi: float | None = None) -> int:
    return _number(v, name, int, lo, hi)


def as_float(
    v: Any, name: str = "option", lo: float | None = None, hi: float | None = None
) -> float:
    return _number(v, name, float, lo, hi)


def optional(conv, v: Any, *args: Any, **kw: Any) -> Any:
    """`conv(v, ...)`, or None for None / '' / 'none' / 'null'."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "none", "null")):
        return None
    return conv(v, *args, **kw)


def as_color(v: Any, name: str = "option") -> tuple[int, int, int]:
    """[r, g, b] (or its JSON string, or '#rrggbb') with every channel in 0..255."""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("#") and len(s) == 7:
            try:
                return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
            except ValueError:
                pass
        else:
            try:
                v = json.loads(s)
            except json.JSONDecodeError:
                pass
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"{name} must be [r, g, b] or '#rrggbb', not {v!r}")
    return tuple(as_int(c, f"{name} channel", 0, 255) for c in v)  # type: ignore[return-value]


def warn_unknown(what: str, extra: dict[str, Any]) -> None:
    """Methods that take no such option ignore it, but say so: a typo must not pass silently."""
    if extra:
        log.warning("%s ignores unknown option(s): %s", what, ", ".join(sorted(extra)))
