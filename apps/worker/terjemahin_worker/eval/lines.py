"""Translation-only evaluation on a set of single lines with references (no page, no OCR).

page01 has 13 bubbles, and the text rules in translate/textproc.py were developed against it,
so its scores flatter the translators. `eval/fixtures/lines_ja_id.tsv` holds held-out lines;
each translator is scored with the same meaning test as the page benchmark: embedding cosine
>= THRESHOLDS["semantic_sim"] to the reference and no metrics.meaning_conflicts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import metrics as M


@dataclass
class Line:
    ja: str
    ref: str
    note: str = ""


def load_lines(path: Path) -> list[Line]:
    out = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) < 2:
            raise ValueError(f"{path}: expected 'ja<TAB>id_ref[<TAB>note]', got {raw!r}")
        out.append(Line(cols[0].strip(), cols[1].strip(), cols[2].strip() if len(cols) > 2 else ""))
    return out


def score(hyps: list[str], lines: list[Line], sim_min: float) -> dict:
    from .semantic import get_embedder

    refs = [ln.ref for ln in lines]
    emb = get_embedder()
    sim = (
        np.diag(np.asarray(emb.similarity(hyps, refs), np.float64))
        if emb is not None
        else np.full(len(hyps), np.nan)
    )
    rows = []
    for i, (h, ln) in enumerate(zip(hyps, lines, strict=True)):
        conf = M.meaning_conflicts(h, ln.ref)
        s = float(sim[i])
        ok = bool(h.strip()) and not M.is_japanese(h) and s >= sim_min and not conf
        rows.append({"i": i, "ja": ln.ja, "ref": ln.ref, "hyp": h, "sim": round(s, 3),
                     "conflicts": conf, "adequate": ok, "note": ln.note})
    return {
        "adequate": sum(r["adequate"] for r in rows),
        "n": len(rows),
        "chrf": M.chrf(hyps, refs),
        "conflicts": sum(bool(r["conflicts"]) for r in rows),
        "semantic": emb is not None,
        "rows": rows,
    }


def run(lines_path: Path, translators: list[str], out: Path | None, settings) -> dict:
    from ..models import BBox, TextRegion
    from ..registry import REGISTRIES, load_all
    from ..translate import TranslationRequest
    from .bench import THRESHOLDS

    load_all()
    lines = load_lines(lines_path)
    page = np.full((64, 64, 3), 255, np.uint8)
    regions = [TextRegion(id=i, bbox=BBox(x1=0, y1=0, x2=10, y2=10)) for i in range(len(lines))]
    report: dict = {"lines": str(lines_path), "threshold": THRESHOLDS["semantic_sim"], "runs": {}}
    for name in translators:
        tr = REGISTRIES["translator"].build(name, settings)
        t = time.perf_counter()
        items = tr.translate(
            TranslationRequest(page_bgr=page, regions=regions, sources=[ln.ja for ln in lines])
        )
        secs = time.perf_counter() - t
        res = score([it.target for it in items], lines, THRESHOLDS["semantic_sim"])
        res["seconds"] = round(secs, 1)
        report["runs"][name] = res
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "lines.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
        (out / "lines.md").write_text(markdown(report))
    return report


def markdown(report: dict) -> str:
    names = list(report["runs"])
    L = [
        f"# Line translation check ({report['lines']})",
        "",
        (f"adequate = cosine >= {report['threshold']} to the reference and no meaning conflict "
        "(negation / participants / roles / voice)."),
        "",
        "| translator | adequate | conflicts | chrF | seconds |",
        "|---|---|---|---|---|",
    ]
    for n in names:
        r = report["runs"][n]
        L.append(f"| {n} | {r['adequate']}/{r['n']} | {r['conflicts']} | {r['chrf']:.1f} | {r['seconds']} |")
    L += ["", "| # | ja | reference | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for i in range(len(report["runs"][names[0]]["rows"]) if names else 0):
        row0 = report["runs"][names[0]]["rows"][i]
        cells = []
        for n in names:
            r = report["runs"][n]["rows"][i]
            mark = "ok" if r["adequate"] else "NO"
            extra = f" [{'/'.join(r['conflicts'])}]" if r["conflicts"] else ""
            cells.append(f"{r['hyp']} ({mark} {r['sim']}{extra})")
        L.append(f"| {i} | {row0['ja']} | {row0['ref']} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"
