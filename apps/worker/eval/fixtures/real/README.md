# Real pages beyond page01

`page01.png` (one level up) is the reference project's own demo page: clean, colour, digital
lettering, no furigana. These two pages are held-out real B/W pages for the eval harness:

| file | source | licence | what is hard |
|---|---|---|---|
| `psimera_val_000002.jpg` | PSImera/manga_bubbles_detect, `images/val` | CC BY-NC 4.0 | 640 px resize, small soft text over dark art, a 3-column balloon |
| `psimera_val_000005.jpg` | PSImera/manga_bubbles_detect, `images/val` | CC BY-NC 4.0 | 640 px resize, furigana on most kanji, a vertical margin disclaimer |

Neither is in the split the shipped YOLO model was trained on (it used `images/test` and
augmented views of page01). The GT files are hand transcriptions with hand-written
Indonesian references; non-dialogue text is listed under `sfx` and not scored.

```sh
.venv/bin/python -m terjemahin_worker.eval bench --page eval/fixtures/real/psimera_val_000005.jpg \
    --gt eval/fixtures/real/psimera_val_000005.gt.json --out out/bench_psi5
```

Two further B/W spreads with GT were used for the evidence in docs/PROOF.md (Manga109
`ARMS` p.15 and `Arisa` p.10, transcribed by a reviewer). Manga109 may not be redistributed,
so only their GT files are kept, in `eval/heldout/`; the images come from Manga109 itself.
