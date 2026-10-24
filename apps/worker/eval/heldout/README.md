# Held-out GT for Manga109 pages (images not included)

`ARMS_015.gt.json` and `Arisa_010.gt.json` are a reviewer's hand transcriptions (dialogue in
`bubbles`, sound effects in `sfx`, no Indonesian references) of two Manga109 two-page B/W
spreads, 1654x1170: `ARMS` page 15 and `Arisa` page 10. Manga109 is licensed for academic
use and may not be redistributed, so the images are not in this repository; with Manga109
access, put the page as `<name>.jpg` next to its GT and run

```sh
.venv/bin/python -m terjemahin_worker.eval bench --page eval/heldout/ARMS_015.jpg \
    --gt eval/heldout/ARMS_015.gt.json --out out/bench_arms
```

Without `id_ref` the translate check reports chrF 0 and skips the meaning test; detection,
OCR, inpainting and rendering are scored as on any page.
