import json, sys, time
from pathlib import Path
from terjemahin_worker.config import Settings
from terjemahin_worker.registry import load_all
from terjemahin_worker.eval.lines import load_lines, score, Line
from terjemahin_worker.eval.bench import THRESHOLDS
regs = load_all()
held = load_lines(Path("eval/fixtures/lines_ja_id.tsv"))
gt = json.load(open("eval/fixtures/page01.gt.json"))["bubbles"]
page = [Line(ja=b["ja"], ref=b["id_ref"], note="") for b in gt]
variants = json.loads(sys.argv[1])
for name, opts in variants.items():
    tr = regs["translator"].build(opts.pop("_t", "hymt"), Settings(), **opts)
    for label, ls in (("page01", page), ("held", held)):
        t = time.time(); hyps = tr.translate_texts([l.ja for l in ls], "ja", "id")
        r = score(hyps, ls, THRESHOLDS["semantic_sim"])
        print(f"{name:14} {label:7} adequate {r['adequate']}/{r['n']} chrF {r['chrf']:.1f} conflicts {r['conflicts']} {time.time()-t:.0f}s", flush=True)
        if label == "page01" and name.startswith("dump"):
            for l, h, row in zip(ls, hyps, r["rows"]): print("   ", "OK " if row["adequate"] else "-- ", l.ja, "=>", h)
