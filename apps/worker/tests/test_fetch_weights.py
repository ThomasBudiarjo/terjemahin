"""`terjemahin-worker fetch-weights`: offline, with files that are already present (no network)."""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from terjemahin_worker import cli
from terjemahin_worker.config import Settings
from terjemahin_worker.weights import WEIGHTS


@pytest.fixture
def weights_dir(tmp_path, monkeypatch):
    for spec in WEIGHTS.values():
        (tmp_path / spec.filename).write_bytes(b"x")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("TERJEMAHIN_OFFLINE", "1")
    return tmp_path


def run(*args):
    return CliRunner().invoke(cli.app, ["fetch-weights", *args])


def test_all_present_is_a_no_op(weights_dir):
    r = run("--all", "--no-hf")
    assert r.exit_code == 0, r.output
    for key in WEIGHTS:
        assert f"present  weight {key} " in r.output
    assert "fetched" not in r.output


def test_missing_file_offline_fails_and_names_it(weights_dir):
    (weights_dir / WEIGHTS["lama"].filename).unlink()
    r = run("lama", "--no-hf")
    assert r.exit_code == 1
    assert "MISSING  weight lama" in r.output


def test_defaults_cover_every_stage_and_the_non_japanese_ocr():
    keys, methods = cli._fetch_targets([], False, Settings())
    s = Settings()
    assert {(st, getattr(s, st)) for st in ("detector", "ocr", "translator")} <= methods
    assert ("ocr", s.ocr_non_japanese) in methods
    if s.detector == "ctd" and s.inpainter == "lama_manga":
        assert {"ctd", "lama_manga"} <= keys


def test_targets_resolve_keys_methods_and_stage_prefix():
    keys, methods = cli._fetch_targets(["lama", "48px", "ocr:none", "inpaint=lama_manga"], False,
                                       Settings())
    assert {"lama", "ocr_48px", "ocr_48px_alphabet", "lama_manga"} <= keys
    assert {("ocr", "48px"), ("ocr", "none"), ("inpainter", "lama_manga")} <= methods
    with pytest.raises(typer.BadParameter, match="stage:none"):
        cli._fetch_targets(["none"], False, Settings())  # ocr and translator both have one
    with pytest.raises(typer.BadParameter, match="neither a weight key"):
        cli._fetch_targets(["bogus"], False, Settings())
    with pytest.raises(ValueError, match="unknown ocr"):
        cli._fetch_targets(["ocr:bogus"], False, Settings())


def test_hf_models_of_selected_methods(weights_dir, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_hf_fetch", lambda repo, offline: (seen.append((repo, offline))
                                                                  or (weights_dir, False)))
    r = run("local_llm", "manga_ocr_hf")
    assert r.exit_code == 0, r.output
    assert sorted(repo for repo, _ in seen) == [
        "Qwen/Qwen2.5-3B-Instruct", "facebook/m2m100_418M", "facebook/nllb-200-distilled-600M",
        "kha-white/manga-ocr-base",
    ]
    assert all(offline for _, offline in seen)
    assert run("local_llm", "--no-hf").exit_code == 0 and len(seen) == 4
    assert cli._hf_models("translator", "chain") == [  # default steps m2m100:en,nllb:id
        "facebook/m2m100_418M", "facebook/nllb-200-distilled-600M"]


def test_local_only_detector_weights_are_reported(weights_dir, monkeypatch):
    monkeypatch.setenv("YOLO_WEIGHTS", str(weights_dir / "nope.onnx"))
    r = run("yolo")
    assert r.exit_code == 1
    assert "MISSING  local  yolo_weights" in r.output
