from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from terjemahin_worker.config import WORKER_ROOT, Settings
from terjemahin_worker.detect import Detection
from terjemahin_worker.models import BBox, TextRegion
from terjemahin_worker.ocr.base import OcrResult
from terjemahin_worker.registry import DETECTORS, INPAINTERS, MASK_REFINERS, OCRS, load_all
from terjemahin_worker.web.app import create_app

load_all()

BOXES = [BBox(x1=300, y1=40, x2=360, y2=200), BBox(x1=60, y1=60, x2=120, y2=220)]
CALLS: dict[str, list] = {"inpaint": [], "raw_mask": []}


class FakeDetector:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def detect(self, img: np.ndarray) -> Detection:
        if self.fail:
            raise RuntimeError("detector exploded")
        mask = np.zeros(img.shape[:2], np.uint8)
        for b in BOXES:
            mask[b.y1 + 10 : b.y2 - 10, b.x1 + 10 : b.x2 - 10] = 255
        return Detection([TextRegion(id=0, bbox=b) for b in BOXES], mask)


class FakeOCR:
    def recognize(self, img, regions):
        return [OcrResult(text=f"テキスト{r.bbox.x1}", confidence=0.9) for r in regions]


class RecordingMask:
    def refine(self, img, regions, raw_mask):
        CALLS["raw_mask"].append(raw_mask is not None)
        m = np.zeros(img.shape[:2], np.uint8)
        for r in regions:
            m[r.bbox.y1 : r.bbox.y2, r.bbox.x1 : r.bbox.x2] = 255
        return m


class CountingFill:
    def inpaint(self, img, mask):
        CALLS["inpaint"].append(int((mask > 0).sum()))
        out = img.copy()
        out[mask > 0] = 255
        return out


def _register() -> None:
    fakes = [
        (DETECTORS, "_web_fake", lambda s, fail=False, **_: FakeDetector(fail)),
        (OCRS, "_web_fake", lambda s, **_: FakeOCR()),
        (MASK_REFINERS, "_web_fake", lambda s, **_: RecordingMask()),
        (INPAINTERS, "_web_fake", lambda s, **_: CountingFill()),
    ]
    for reg, name, factory in fakes:
        if name not in reg.methods:
            reg.register(name, description="test fake")(factory)


_register()

CHEAP = {
    "detector": "_web_fake",
    "ocr": "_web_fake",
    "translator": "original",
    "mask_refiner": "_web_fake",
    "inpainter": "_web_fake",
    "renderer": "basic",
}


def page_png() -> bytes:
    img = np.full((260, 420, 3), 255, np.uint8)
    for b in BOXES:
        cv2.rectangle(img, (b.x1 + 15, b.y1 + 15), (b.x2 - 15, b.y2 - 15), (0, 0, 0), -1)
    return cv2.imencode(".png", img)[1].tobytes()


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None
    return img


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(storage_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        yield c


def post_page(client: TestClient, pipeline: dict | None = None, **form) -> dict:
    data = {"pipeline": json.dumps(pipeline or CHEAP), **form}
    r = client.post("/api/translate", files={"file": ("p.png", page_png(), "image/png")}, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_index_is_self_contained(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    html = r.text
    assert "/api/methods" in html and "/api/compose" in html and "/api/translate" in html
    assert not re.search(r"<(script|link|img)[^>]+(src|href)=\"https?://", html)


def test_methods_lists_every_stage_with_availability(client):
    m = client.get("/api/methods").json()
    assert set(m["stages"]) == {"detector", "ocr", "translator", "mask_refiner", "inpainter",
                                "renderer"}
    for stage, info in m["stages"].items():
        names = [r["name"] for r in info["methods"]]
        assert info["default"] in names, stage
        assert all({"available", "reason", "description"} <= set(r) for r in info["methods"])
    tr = {r["name"]: r for r in m["stages"]["translator"]["methods"]}
    assert tr["original"]["available"]


def test_translate_returns_result_and_stage_images(client, tmp_path):
    j = post_page(client, context=json.dumps({"glossary": [{"source": "a", "target": "b"}]}))
    assert j["status"] == "done", j
    assert [b["source_text"] for b in j["result"]["bubbles"]] == ["テキスト300", "テキスト60"]
    assert j["result"]["pipeline"]["renderer"] == "basic"
    assert j["context"]["glossary"][0]["target"] == "b"
    assert set(j["result"]["timings"]) >= {"detect", "ocr", "translate", "inpaint", "render"}
    for name in ("original", "detect", "mask", "inpainted", "final"):
        r = client.get(j["images"][name])
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", name
        assert decode(r.content).shape[:2] == (260, 420)
    final = decode(client.get(j["images"]["final"]).content)
    assert final[50:190, 315:345].mean() > 150  # black block erased, text drawn on white
    d = tmp_path / "web" / j["id"]
    assert (d / "raw_mask.png").exists() and (d / "result.json").exists()
    dl = client.get(j["download"])
    assert dl.status_code == 200 and "attachment" in dl.headers["content-disposition"]


def test_async_job_reports_progress_until_done(client):
    j = post_page(client, wait="false")
    assert j["status"] in {"queued", "running", "done"}
    seen = set()
    for _ in range(200):
        j = client.get(f"/api/jobs/{j['id']}").json()
        seen.add(j["status"])
        if j["status"] == "done":
            break
        time.sleep(0.02)
    assert j["status"] == "done" and j["stage"] is None


def test_compose_rerenders_only_when_erase_set_unchanged(client, tmp_path):
    j = post_page(client)
    CALLS["inpaint"].clear()
    CALLS["raw_mask"].clear()
    bubbles = j["result"]["bubbles"]
    before = decode(client.get(j["images"]["final"]).content)

    bubbles[0]["target_text"] = "HALO DUNIA PANJANG SEKALI"
    r = client.post("/api/compose", json={"job_id": j["id"], "bubbles": bubbles})
    assert r.status_code == 200, r.text
    j2 = r.json()
    assert CALLS["inpaint"] == []  # text-only edit: stored inpainted page reused
    assert j2["version"] == j["version"] + 1 and j2["images"]["final"] != j["images"]["final"]
    assert j2["result"]["bubbles"][0]["target_text"] == "HALO DUNIA PANJANG SEKALI"
    after = decode(client.get(j2["images"]["final"]).content)
    assert np.abs(after.astype(int) - before.astype(int)).sum() > 0
    saved = json.loads((tmp_path / "web" / j["id"] / "result.json").read_text())
    assert saved["bubbles"][0]["target_text"] == "HALO DUNIA PANJANG SEKALI"

    # dropping a bubble changes what is erased: mask + inpaint run again, with the raw mask
    r = client.post(
        "/api/compose",
        json={"job_id": j["id"], "bubbles": bubbles[:1], "dropped": bubbles[1:]},
    )
    assert r.status_code == 200, r.text
    assert [b["region"]["id"] for b in r.json()["dropped"]] == [1]
    assert len(r.json()["result"]["bubbles"]) == 1
    assert len(CALLS["inpaint"]) == 1 and CALLS["raw_mask"] == [True]
    final = decode(client.get(r.json()["images"]["final"]).content)
    assert final[90:170, 85:95].mean() < 50  # the dropped bubble keeps its original ink


def test_compose_can_switch_renderer(client):
    j = post_page(client)
    r = client.post(
        "/api/compose",
        json={"job_id": j["id"], "bubbles": j["result"]["bubbles"],
              "pipeline": {"renderer": "default"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["pipeline"]["renderer"] == "default"
    assert r.json()["pipeline"]["detector"] == "_web_fake"


def test_bad_requests(client):
    files = {"file": ("p.png", page_png(), "image/png")}
    r = client.post("/api/translate", files=files, data={"pipeline": "{not json"})
    assert r.status_code == 400
    r = client.post("/api/translate", files=files,
                    data={"pipeline": json.dumps({**CHEAP, "detector": "nope"})})
    assert r.status_code == 400 and "unknown detector" in r.json()["detail"]
    r = client.post("/api/translate", files=files,
                    data={"pipeline": json.dumps({**CHEAP, "translator": "openai"})})
    assert r.status_code == 400 and "unavailable" in r.json()["detail"]
    r = client.post("/api/translate", files={"file": ("x.png", b"nope", "image/png")})
    assert r.status_code == 400
    r = client.post("/api/translate", files=files,
                    data={"pipeline": json.dumps({**CHEAP, "options": {"bogus": {}}})})
    assert r.status_code == 400
    assert client.get("/api/jobs/0123456789abcdef").status_code == 404
    assert client.get("/api/jobs/..%2F..").status_code == 404
    j = post_page(client)
    assert client.get(f"/api/jobs/{j['id']}/files/job.json").status_code == 404
    assert client.get(f"/api/jobs/{j['id']}/files/..%2Fx.png").status_code == 404
    r = client.post("/api/compose", json={"job_id": "0123456789abcdef", "bubbles": []})
    assert r.status_code == 404


def test_failed_job_reports_error(client):
    j = post_page(client, {**CHEAP, "options": {"detector": {"fail": True}}})
    assert j["status"] == "failed" and "detector exploded" in j["error"]
    r = client.post("/api/compose", json={"job_id": j["id"], "bubbles": []})
    assert r.status_code == 409


def test_jobs_survive_restart(tmp_path):
    settings = Settings(storage_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        j = post_page(c)
    with TestClient(create_app(settings)) as c:
        again = c.get(f"/api/jobs/{j['id']}").json()
        assert again["status"] == "done" and again["result"] == j["result"]
        assert c.get(again["images"]["final"]).status_code == 200


def test_cli_has_web_command():
    from typer.testing import CliRunner

    from terjemahin_worker.cli import app

    r = CliRunner().invoke(app, ["web", "--help"])
    assert r.exit_code == 0 and "--port" in r.output


PAGE = WORKER_ROOT / "eval" / "fixtures" / "page01.png"
REAL = ["koharu-comictextdetector.onnx", "koharu-manga-ocr.onnx", "koharu-lama-manga.onnx"]


@pytest.mark.skipif(
    not PAGE.exists() or not all((WORKER_ROOT / "weights" / w).exists() for w in REAL),
    reason="needs page01 and ctd / manga-ocr / lama-manga weights",
)
def test_real_methods_on_page01(tmp_path):
    """ctd + manga_ocr + default mask + lama_manga + default renderer through the API."""
    cfg = {"detector": "ctd", "ocr": "manga_ocr", "translator": "original",
           "mask_refiner": "default", "inpainter": "lama_manga", "renderer": "default"}
    with TestClient(create_app(Settings(storage_dir=tmp_path))) as c:
        r = c.post("/api/translate", files={"file": ("page01.png", PAGE.read_bytes(), "image/png")},
                   data={"pipeline": json.dumps(cfg)})
        j = r.json()
        assert j["status"] == "done", j
        bubbles = j["result"]["bubbles"]
        assert len(bubbles) >= 10
        assert sum(len(b["source_text"]) >= 3 for b in bubbles) >= 8
        for b in bubbles:
            b["target_text"] = "Halo!"
        t = time.perf_counter()
        r = c.post("/api/compose", json={"job_id": j["id"], "bubbles": bubbles})
        assert r.status_code == 200, r.text
        assert time.perf_counter() - t < 5  # render-only path, no re-inpaint
        final = decode(c.get(r.json()["images"]["final"]).content)
        assert final.shape[:2] == (2048, 1487)
