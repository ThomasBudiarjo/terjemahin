import numpy as np

from terjemahin_worker.clean import fill_region
from terjemahin_worker.detect import sort_reading_order
from terjemahin_worker.models import BBox, TextRegion


def test_fill_region_uses_surrounding_colour():
    img = np.full((100, 100, 3), 240, dtype=np.uint8)  # light bubble
    img[40:60, 40:60] = 0  # black "text"
    color = fill_region(img, BBox(x1=40, y1=40, x2=60, y2=60))
    assert color == (240, 240, 240)
    assert (img[40:60, 40:60] == 240).all()


def test_reading_order_right_to_left_then_top_to_bottom():
    r = lambda i, x, y: TextRegion(id=i, bbox=BBox(x1=x, y1=y, x2=x + 40, y2=y + 40))  # noqa: E731
    regions = [r(0, 10, 10), r(1, 200, 12), r(2, 10, 200), r(3, 200, 205)]
    ordered = sort_reading_order(regions)
    xs_ys = [(b.bbox.x1, b.bbox.y1) for b in ordered]
    assert xs_ys == [(200, 12), (10, 10), (200, 205), (10, 200)]
    assert [b.id for b in ordered] == [0, 1, 2, 3]  # ids reassigned to reading order
