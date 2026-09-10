from __future__ import annotations

from typing import Protocol

import numpy as np

from ..models import TextRegion


class Detector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[TextRegion]:
        """Return text regions in the image. IDs are assigned by the caller."""
        ...


def sort_reading_order(regions: list[TextRegion], row_tolerance: float = 0.5) -> list[TextRegion]:
    """Manga reading order: right-to-left, top-to-bottom.

    Regions whose vertical centres are within `row_tolerance * median_height` of each other
    are treated as one row and ordered right-to-left; rows are ordered top-to-bottom.
    Approximate on purpose: panel layout is not known here, and the LLM also sees the
    whole page so small ordering errors are recoverable.
    """
    if not regions:
        return []
    heights = sorted(r.bbox.h for r in regions)
    median_h = heights[len(heights) // 2] or 1
    tol = row_tolerance * median_h

    rows: list[list[TextRegion]] = []
    for r in sorted(regions, key=lambda r: r.bbox.cy):
        if rows and abs(rows[-1][0].bbox.cy - r.bbox.cy) <= tol:
            rows[-1].append(r)
        else:
            rows.append([r])

    ordered: list[TextRegion] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda r: -r.bbox.cx))
    return [TextRegion(id=i, bbox=r.bbox, score=r.score) for i, r in enumerate(ordered)]
