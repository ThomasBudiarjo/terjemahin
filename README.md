# terjemahin

Translate manga pages Japanese → Indonesian, with LLM translation that uses series context
and automatic typesetting back into the bubbles.

```
web  →  apps/api (Go)  →  LavinMQ  →  apps/worker (Python)
            │                              │
        Postgres/store  ←── results ───────┘
```

## Pipeline (worker)

| Stage | Implementation | Swap point |
|---|---|---|
| Detect text regions | YOLO11n fine-tuned on Manga109 (ONNX, CPU ~100ms) or comic-text-detector fallback | `detect/` `Detector` protocol |
| Read + translate | Gemini vision: whole page + per-bubble crops + glossary + previous pages → JSON | `translate/` `Translator` protocol |
| Clean | fill box with surrounding background colour | `clean.py` |
| Typeset | Pillow, auto font size, word wrap, centred, contrast-aware colour | `typeset.py` |

Out of scope for v1: SFX outside bubbles, inpainting, vertical Indonesian text.

## Worker quick start

```sh
cd apps/worker
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -e ".[dev]"
cp ../../.env.example .env        # fill GEMINI_API_KEY

# detector weights: see weights/README.md (CTD fallback download or train YOLO)
DETECTOR=ctd .venv/bin/terjemahin-worker detect page.jpg          # debug boxes
DETECTOR=ctd .venv/bin/terjemahin-worker translate page.jpg -o out/page.png
.venv/bin/terjemahin-worker compose page.jpg out/page.json -o out/page.png   # re-typeset after manual edit
.venv/bin/python -m pytest
```

### Train the detector (RTX 4060)

```sh
uv pip install -p .venv/bin/python -e ".[train]"
python training/prepare_manga109.py --root /path/to/Manga109 --out data/manga109-text
python training/train.py --data data/manga109-text/dataset.yaml --epochs 60
# -> weights/manga-text-yolo11n.onnx
```

## API quick start

```sh
docker compose up lavinmq postgres -d
cd apps/api && go run .
curl -F file=@page.jpg -F 'context={"glossary":[{"source":"ナルト","target":"Naruto"}]}' localhost:8080/v1/pages
curl localhost:8080/v1/jobs/<id>            # status
curl localhost:8080/v1/jobs/<id>/image -o out.png
```

Job store is in-memory for now; Postgres wiring is the next step.

## Deploy budget (Railway)

- `api`: ~64MB RAM, always on.
- `worker`: ~1GB RAM peak, enable sleep-on-idle. CPU inference only.
- LavinMQ: CloudAMQP free tier, or self-host via compose.
