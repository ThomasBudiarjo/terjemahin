"""Detect + OCR only, scored against GT text with metrics.match_detections (and coverage).

usage: deteval.py OUTDIR DET page.jpg:gt.json [...]   (DET may carry opts: ctd or ctd,tile=1)
"""
import json
import os
import sys
import time

import cv2

from terjemahin_worker.config import get_settings
from terjemahin_worker.detect import sort_reading_order
from terjemahin_worker.eval import metrics as M
from terjemahin_worker.models import PipelineConfig
from terjemahin_worker.pipeline import Pipeline, draw_regions

out = sys.argv[1]
det = sys.argv[2]
opts = {}
if "," in det:
    det, *kv = det.split(",")
    opts = dict(x.split("=", 1) for x in kv)
os.makedirs(out, exist_ok=True)
pipe = Pipeline(get_settings())
cfg = pipe.resolve(PipelineConfig(detector=det, options={"detector": opts}))
tot = {"gt": 0, "match": 0, "fp": 0, "dup": 0, "partial": 0}
for spec in sys.argv[3:]:
    page, gtp = spec.split(":")
    gt = json.load(open(gtp))
    img = cv2.imread(page)
    t = time.time()
    d, regs = pipe.detect(img, cfg)
    td = time.time() - t
    ocr = pipe.stage("ocr", cfg).recognize(img, regs)
    texts = [o.text for o in ocr]
    gts = [b["ja"] for b in gt["bubbles"]]
    sc = M.match_detections(texts, gts, gt.get("sfx", []), boxes=[r.bbox for r in regs])
    name = os.path.splitext(os.path.basename(page))[0]
    cv2.imwrite(f"{out}/{name}_detect.jpg", draw_regions(img, regs), [cv2.IMWRITE_JPEG_QUALITY, 85])
    partial = []
    for m in sc.matches:
        h = M.normalize_ja(texts[m.det], punct=False)
        r = M.normalize_ja(gts[m.gt], punct=False)
        cov = getattr(m, "coverage", None)
        if len(h) < len(r) and m.cer_text > 0:
            partial.append((m.det, texts[m.det], gts[m.gt]))
    print(f"== {name} det {td:.2f}s: {len(sc.matches)}/{sc.n_gt} fp {len(sc.false_positives)} "
          f"dup {len(sc.duplicates)} frag {len(sc.fragments)} sfx {len(sc.sfx)} cer {sc.mean_cer}")
    for i, (r, tx) in enumerate(zip(regs, texts)):
        tag = ""
        if i in sc.false_positives:
            tag = "FP"
        elif i in sc.duplicates:
            tag = "DUP"
        print(f"  {i:2d} {r.bbox.x1},{r.bbox.y1},{r.bbox.x2},{r.bbox.y2} {tx} {tag}")
    print("  missed:", [gts[g] for g in sc.missed])
    print("  shorter-than-GT:", partial)
    tot["gt"] += sc.n_gt
    tot["match"] += len(sc.matches)
    tot["fp"] += len(sc.false_positives)
    tot["dup"] += len(sc.duplicates)
    tot["partial"] += len(partial)
print("TOTAL", tot)
