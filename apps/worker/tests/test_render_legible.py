"""Renderer: a legible size floor (spill with an outline instead of tiny type), text kept on
the page, margin columns set down the column, balloons whose interior misses the text."""

from pathlib import Path

import cv2
import numpy as np

from terjemahin_worker.config import Settings
from terjemahin_worker.models import BBox, Bubble, TextRegion
from terjemahin_worker.registry import RENDERERS, load_all
from terjemahin_worker.render import default as D

FONT = Path(__file__).resolve().parent.parent / "fonts" / "ComicShanns2.ttf"


def renderer():
    load_all()
    return RENDERERS.build("default", Settings(font_path=FONT))


def bubble(x1, y1, x2, y2, text, font=None, direction="v"):
    r = TextRegion(id=0, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), font_size=font,
                   direction=direction)
    return Bubble(region=r, source_text="あ", target_text=text)


def test_legible_size():
    assert D.legible_size(14) == 13  # a 1149 px page: no longer 0.65 x 14 = 9
    assert D.legible_size(18) == 16
    assert D.legible_size(10) == 10  # never above the page minimum


def test_tall_narrow_balloon_is_not_set_in_tiny_type():
    # judge 3: 'Diamlah!' at 11 px in a 45x200 px balloon of a 1149 px page
    page = np.full((1678, 1149, 3), 120, np.uint8)
    cv2.ellipse(page, (500, 600), (24, 100), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (500, 600), (24, 100), 0, 0, 360, (0, 0, 0), 2)
    r = renderer()
    r.render(page, [bubble(488, 540, 512, 660, "Diamlah!", font=20)], page)
    (p,) = r.plans
    assert p.layout.size >= D.legible_size(D.page_min_size(1149, 1678))


def test_text_that_cannot_fit_spills_with_an_outline_and_stays_on_the_page():
    page = np.full((400, 300, 3), 120, np.uint8)
    cv2.ellipse(page, (28, 360), (26, 38), 0, 0, 360, (255, 255, 255), -1)  # at the page edge
    cv2.ellipse(page, (28, 360), (26, 38), 0, 0, 360, (0, 0, 0), 2)
    r = renderer()
    out = r.render(page, [bubble(10, 335, 46, 385, "Selanjutnya dan seterusnya ya")], page)
    (p,) = r.plans
    assert not p.layout.fits and p.layout.size >= 10
    assert r.colors(p, p.layout.size)[2] > 0  # an outline: it crosses the balloon's
    ink = np.abs(out.astype(int) - page.astype(int)).max(axis=2) > 40
    ys, xs = np.nonzero(ink)
    assert xs.min() >= 0 and ys.max() < 400
    assert ink[:, :2].sum() < 0.01 * ink.sum()  # not run off the left edge


def test_margin_column_is_set_down_the_column():
    # judge 4: the vertical margin note was set across a widened box, over the panel art
    rng = np.random.default_rng(0)
    page = rng.integers(60, 255, (640, 640, 1), dtype=np.uint8).repeat(3, axis=2)  # art
    text = "Kisah ini adalah fiksi, dan tidak ada hubungannya dengan orang sungguhan."
    r = renderer()
    r.render(page, [bubble(611, 310, 624, 601, text)], page)
    (p,) = r.plans
    assert p.angle == -90.0
    assert D._outside(p, BBox(x1=611, y1=310, x2=624, y2=601)) < 60 * 291


def test_balloon_whose_interior_misses_the_text_is_not_used():
    from terjemahin_worker.render.balloon import Balloon

    mask = np.zeros((300, 100), np.uint8)
    mask[0:20, 80:100] = 255  # a sliver far from the text box
    bal = Balloon(mask, 0, 0, True, (255, 255, 255))
    assert not D._holds(bal, BBox(x1=10, y1=100, x2=30, y2=280))
    mask[:, :] = 255
    assert D._holds(bal, BBox(x1=10, y1=100, x2=30, y2=280))


def test_tall_column_inside_the_page_keeps_horizontal_text():
    rng = np.random.default_rng(0)
    page = rng.integers(60, 255, (640, 640, 1), dtype=np.uint8).repeat(3, axis=2)
    r = renderer()
    r.render(page, [bubble(311, 310, 324, 601, "Tapi ya ya ya, aku tahu semuanya")], page)
    (p,) = r.plans
    assert p.angle == 0.0
