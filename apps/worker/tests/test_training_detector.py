"""Detector training path: prune_ctd -> ctd_blk, pseudo_label, train.py (YOLO) -> `-d yolo`."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from terjemahin_worker.config import Settings
from terjemahin_worker.detect import DETECTORS, Detection
from terjemahin_worker.models import BBox, TextRegion

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import prune_ctd
import pseudo_label
import train

SETTINGS = Settings()
CTD_FULL = SETTINGS.weights_dir / "koharu-comictextdetector.onnx"
YOLO_ONNX = SETTINGS.yolo_weights
PAGE = ROOT / "eval" / "fixtures" / "page01.png"
# centres of the 13 dialogue bubbles on page01 (same list as test_detect_ctd.py)
BUBBLES = [
    (1387, 160), (232, 98), (1184, 276), (68, 281), (1400, 569), (649, 985), (63, 1186),
    (1444, 1554), (1106, 1536), (335, 1604), (908, 1652), (724, 1710), (88, 1870),
]


def covered(regions, pts=BUBBLES) -> int:
    return sum(
        any(r.bbox.x1 <= x <= r.bbox.x2 and r.bbox.y1 <= y <= r.bbox.y2 for r in regions)
        for x, y in pts
    )


def boxes(det: Detection) -> list[tuple[int, int, int, int]]:
    return sorted((r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2) for r in det.regions)


# ------------------------------------------------------------------------------ prune_ctd


def test_prune_default_src_prefers_original_then_koharu(tmp_path, monkeypatch):
    monkeypatch.setattr(prune_ctd, "WEIGHTS", tmp_path)
    assert prune_ctd.default_src() == tmp_path / "comictextdetector.pt.onnx"  # neither: error path
    (tmp_path / "koharu-comictextdetector.onnx").write_bytes(b"")
    assert prune_ctd.default_src().name == "koharu-comictextdetector.onnx"
    (tmp_path / "comictextdetector.pt.onnx").write_bytes(b"")
    assert prune_ctd.default_src().name == "comictextdetector.pt.onnx"


@pytest.mark.skipif(not (CTD_FULL.exists() and PAGE.exists()), reason="ctd weights missing")
def test_pruned_ctd_blk_equals_ctd_blocks_on_page01(tmp_path):
    import onnxruntime as ort

    out = tmp_path / "blk.onnx"
    prune_ctd.main(["--src", str(CTD_FULL), "--out", str(out)])
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["blk"]
    assert out.stat().st_size < CTD_FULL.stat().st_size / 2

    page = cv2.imread(str(PAGE))
    blk = DETECTORS.build("ctd_blk", SETTINGS, weights=str(out)).detect(page)
    raw = DETECTORS.build("ctd", SETTINGS, heads="blk").detect(page)
    assert boxes(blk) == boxes(raw)
    assert blk.mask is None and all(not r.lines for r in blk.regions)
    assert covered(blk.regions) == len(BUBBLES)


# --------------------------------------------------------------------------- pseudo_label


class DarkBoxTeacher:
    """Labels every dark connected blob: the label is exactly where the ink is."""

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, _, stats, _ = cv2.connectedComponentsWithStats((gray < 80).astype(np.uint8))
        regions = [
            TextRegion(id=i, bbox=BBox(x1=int(x), y1=int(y), x2=int(x + w), y2=int(y + h)))
            for i, (x, y, w, h, area) in enumerate(stats[1:].tolist())
            if area > 30
        ]
        return Detection(regions)


def test_pseudo_label_augmented_views_get_teacher_labels(tmp_path, monkeypatch):
    page = np.full((900, 640, 3), 235, np.uint8)
    for x, y in ((80, 100), (420, 300), (200, 650)):
        cv2.rectangle(page, (x, y), (x + 60, y + 160), (0, 0, 0), -1)
    src = tmp_path / "page.png"
    cv2.imwrite(str(src), page)

    class Reg:
        def build(self, name, settings, **opts):
            assert name == "ctd" and opts == {"conf": 0.5}
            return DarkBoxTeacher()

    monkeypatch.setattr(pseudo_label, "DETECTORS", Reg())
    out = tmp_path / "ds"
    stats = pseudo_label.main(
        ["--pool", str(src), "--out", str(out), "--augment", "4", "--val-split", "train",
         "--prefix", "p_", "--review", "1"]
    )
    imgs = sorted((out / "images" / "train").glob("*.jpg"))
    assert [p.stem for p in imgs] == ["p_page", "p_page_a000", "p_page_a001", "p_page_a002",
                                      "p_page_a003"]
    assert stats["images"] == 5 and not list((out / "images" / "val").iterdir())
    sizes = {cv2.imread(str(p)).shape[:2] for p in imgs}
    assert len(sizes) > 1  # views are rescaled / cropped, not copies
    for p in imgs:
        img = cv2.imread(str(p))
        rows = np.loadtxt(out / "labels" / "train" / f"{p.stem}.txt", ndmin=2)
        assert rows.shape[1] == 5 and (rows[:, 0] == 0).all()
        assert ((rows[:, 1:] > 0) & (rows[:, 1:] <= 1)).all()
        # labels follow the view: every box centre sits on ink in that image
        H, W = img.shape[:2]
        for _, cx, cy, _, _ in rows:
            assert img[int(cy * H), int(cx * W)].mean() < 120
    cfg = yaml.safe_load((out / "dataset.yaml").read_text())
    assert cfg["names"] == {0: "text"} and cfg["train"] == "images/train"
    assert len(list((out / "review").glob("*.jpg"))) == 1


def test_pseudo_label_teacher_without_conf_option(tmp_path, monkeypatch):
    page = np.full((300, 200, 3), 235, np.uint8)
    cv2.rectangle(page, (50, 50), (90, 200), (0, 0, 0), -1)
    cv2.imwrite(str(tmp_path / "a.png"), page)

    class Reg:
        def build(self, name, settings, **opts):
            if opts:
                raise TypeError("unexpected keyword argument 'conf'")
            return DarkBoxTeacher()

    monkeypatch.setattr(pseudo_label, "DETECTORS", Reg())
    stats = pseudo_label.main(["--pool", str(tmp_path), "--out", str(tmp_path / "ds"),
                               "--detector", "classical", "--val-ratio", "0"])
    assert stats == {"images": 1, "boxes": 1, "pages": 1}


# ------------------------------------------------------------------------------- train.py


def _dataset(root: Path, n_train: int, n_val: int) -> Path:
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            (root / "images" / split / f"{i}.jpg").write_bytes(b"")
    y = root / "dataset.yaml"
    y.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: text\n")
    return y


def test_merge_datasets_lists_every_nonempty_split(tmp_path):
    a = _dataset(tmp_path / "synth", 3, 1)
    b = _dataset(tmp_path / "pseudo", 2, 0)  # empty val dir is left out
    assert train.merge_datasets([a], tmp_path / "m.yaml") == a
    merged = yaml.safe_load(train.merge_datasets([a, b], tmp_path / "run" / "m.yaml").read_text())
    assert merged["train"] == [str(tmp_path / "synth/images/train"),
                               str(tmp_path / "pseudo/images/train")]
    assert merged["val"] == [str(tmp_path / "synth/images/val")]
    assert merged["names"] == {0: "text"}


def test_resolve_model_keeps_local_and_non_checkpoint_names(tmp_path):
    local = tmp_path / "yolo11n.pt"
    local.write_bytes(b"")
    assert train.resolve_model(str(local)) == str(local)
    assert train.resolve_model("yolo11n.yaml") == "yolo11n.yaml"  # from-scratch config
    assert train.resolve_model("runs/x/best.pt") == "runs/x/best.pt"  # explicit path


def _yolo11n_cached() -> bool:
    if importlib.util.find_spec("ultralytics") is None:
        return False
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(train.HF_MIRROR, "yolo11n.pt"), str)


@pytest.mark.skipif(not _yolo11n_cached(), reason="ultralytics or cached yolo11n.pt missing")
def test_train_one_epoch_exports_onnx_the_worker_loads(tmp_path):
    """train.py end to end at a toy size: merged datasets, 1 epoch, ONNX -> `-d yolo`."""
    rng = np.random.default_rng(0)
    yamls = []
    for name in ("a", "b"):
        root = tmp_path / name
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for i in range(3):
                img = np.full((96, 96, 3), 230, np.uint8)
                x, y = (int(v) for v in rng.integers(5, 50, 2))
                cv2.rectangle(img, (x, y), (x + 30, y + 40), (0, 0, 0), -1)
                cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
                (root / "labels" / split / f"{i}.txt").write_text(
                    f"0 {(x + 15) / 96:.4f} {(y + 20) / 96:.4f} {30 / 96:.4f} {40 / 96:.4f}\n"
                )
        (root / "dataset.yaml").write_text(
            f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: text\n"
        )
        yamls.append(str(root / "dataset.yaml"))

    out = tmp_path / "det.onnx"
    train.main(["--data", *yamls, "--epochs", "1", "--imgsz", "64", "--batch", "2",
                "--device", "cpu", "--workers", "0", "--project", str(tmp_path / "runs"),
                "--name", "t", "--out", str(out)])
    assert out.exists()
    det = DETECTORS.build("yolo", SETTINGS, weights=str(out), conf=0.0001)
    assert det.size == 64
    res = det.detect(np.full((300, 200, 3), 230, np.uint8))  # runs; 1 epoch learns nothing
    assert isinstance(res, Detection)


@pytest.mark.skipif(not (YOLO_ONNX.exists() and PAGE.exists()), reason="no trained yolo weights")
def test_trained_yolo_finds_page01_bubbles():
    """The shipped weights/manga-text-yolo11n.onnx (training/README.md, page01 was in its
    training data as augmented views, so this checks the plumbing, not generalisation)."""
    det = DETECTORS.build("yolo", SETTINGS)
    res = det.detect(cv2.imread(str(PAGE)))
    assert covered(res.regions) >= 11
    assert len(res.regions) <= 20
