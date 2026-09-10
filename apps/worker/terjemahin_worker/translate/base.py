from __future__ import annotations

from typing import Protocol

import numpy as np

from ..models import Bubble, PageContext, TextRegion


class Translator(Protocol):
    def translate(
        self,
        page_bgr: np.ndarray,
        regions: list[TextRegion],
        context: PageContext,
    ) -> list[Bubble]:
        """Read the Japanese text in each region and translate it to Indonesian.

        Must return exactly one Bubble per region, in the same order.
        """
        ...
