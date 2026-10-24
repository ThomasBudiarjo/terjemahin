"""pages.jsonl (synth.py, exact text per block) + corpus.tsv (my Indonesian refs) -> bench GT JSON."""
import json
import sys
from functools import lru_cache
from pathlib import Path

corpus = dict(l.split("\t") for l in Path("corpus.tsv").read_text().splitlines() if l.strip())
ELL = ["・・・", "……", "．．．", "・・", "…"]

def norm(t):
    for e in ELL:
        t = t.replace(e, "…")
    return t

def decompose(text):
    t = norm(text)
    keys = sorted(corpus, key=len, reverse=True)
    @lru_cache(None)
    def go(i):
        if i == len(t):
            return []
        for k in keys:
            if t.startswith(k, i):
                r = go(i + len(k))
                if r is not None:
                    return [k] + r
        return None
    parts = go(0)
    if parts is None:
        raise ValueError(f"cannot decompose {text!r}")
    return parts

out = Path(sys.argv[1]); out.mkdir(exist_ok=True)
for line in Path(sys.argv[2] if len(sys.argv) > 2 else "gen/pages.jsonl").read_text().splitlines():
    rec = json.loads(line)
    bubbles = [{"ja": b["text"], "id_ref": " ".join(corpus[p] for p in decompose(b["text"])),
                "bbox": b["bbox"], "vertical": b["vertical"], "container": b["container"],
                "font_size": b["font_size"]} for b in rec["blocks"]]
    gt = {"image": f"{rec['image']}.jpg", "note": "synthetic page from training/synth.py --seed 7; ja exact, id_ref = hand translation of each corpus line",
          "bubbles": bubbles, "sfx": ["ごろん", "もじ…"]}
    (out / f"{rec['image']}.gt.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1))
    print(rec["image"], rec["width"], rec["height"], len(bubbles), [ (b["ja"], b["container"], b["font_size"]) for b in bubbles])
