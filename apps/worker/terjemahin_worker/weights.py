"""Model weight manager: files are fetched on first use into Settings.weights_dir.

Add a WeightSpec here for any new downloadable file, then reference its key in the method's
`weights=` registration and resolve it with `ensure(key, settings)`.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

HF = "https://huggingface.co"


@dataclass(frozen=True)
class WeightSpec:
    filename: str
    url: str
    note: str = ""


WEIGHTS: dict[str, WeightSpec] = {
    "ctd": WeightSpec(
        "koharu-comictextdetector.onnx",
        f"{HF}/mayocream/koharu/resolve/main/comictextdetector.onnx",
        "comic-text-detector (blk + seg + det heads), 1024x1024 input",
    ),
    "manga_ocr_onnx": WeightSpec(
        "koharu-manga-ocr.onnx",
        f"{HF}/mayocream/koharu/resolve/main/manga-ocr.onnx",
        "manga-ocr VisionEncoderDecoder in one graph: (image 224, token_ids) -> logits",
    ),
    "manga_ocr_vocab": WeightSpec(
        "koharu-vocab.txt", f"{HF}/mayocream/koharu/resolve/main/vocab.txt"
    ),
    "lama_manga": WeightSpec(
        "koharu-lama-manga.onnx",
        f"{HF}/mayocream/koharu/resolve/main/lama-manga.onnx",
        "LaMa fine-tuned on manga, (image 512, mask 512) -> image",
    ),
    "lama": WeightSpec(
        "lama-carve.onnx",
        f"{HF}/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        "big-lama (places2) ONNX, (image 512, mask 512) -> image",
    ),
    "ocr_48px": WeightSpec(
        "ocr_ar_48px.ckpt",
        f"{HF}/zyddnys/manga-image-translator/resolve/main/ocr_ar_48px.ckpt",
        "manga-image-translator 48px autoregressive OCR (torch)",
    ),
    "ocr_48px_alphabet": WeightSpec(
        "alphabet-all-v7.txt",
        f"{HF}/zyddnys/manga-image-translator/resolve/main/alphabet-all-v7.txt",
    ),
}


def ensure(key: str, settings: Settings) -> Path:
    spec = WEIGHTS[key]
    path = Path(settings.weights_dir) / spec.filename
    if path.exists() and path.stat().st_size > 0:
        return path
    if os.environ.get("TERJEMAHIN_OFFLINE"):
        raise FileNotFoundError(f"{path} missing and TERJEMAHIN_OFFLINE is set")
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", spec.url, path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        with urllib.request.urlopen(spec.url, timeout=60) as resp:
            shutil.copyfileobj(resp, tmp, length=1 << 20)
    Path(tmp.name).replace(path)
    return path
