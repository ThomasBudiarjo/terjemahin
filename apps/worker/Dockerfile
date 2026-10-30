# terjemahin worker: takes jobs from the web app (`serve`, default) or web UI (`web --host 0.0.0.0`).
#
#   docker build -t terjemahin-worker apps/worker
#   docker build --build-arg FETCH="--all" -t terjemahin-worker apps/worker    # every weight file
#   docker build --build-arg EXTRAS=web --build-arg FETCH="ctd manga_ocr lama_manga" ...
#
# EXTRAS: pyproject extras. `torch` (CPU wheels) enables the nllb / m2m100 / local_llm
# translators and the 48px / manga_ocr_hf OCRs; `web` the web UI. Measured before any weights:
# 2.2 GB with torch,web, 0.8 GB with web only (then pick a translator that needs no torch).
# FETCH: arguments of `terjemahin-worker fetch-weights` run at build time. Empty = the default
# method of every stage + the non-Japanese OCR (48px): ~1.0 GB of ONNX/ckpt in /app/weights
# and NLLB-600M (~2.5 GB) in $HF_HOME. Anything not baked in downloads on first use.
# CJK_FONTS=1 installs fonts-noto-cjk for the `vertical` renderer (ja/zh/ko targets).
FROM python:3.12-slim

ARG EXTRAS=torch,web
ARG FETCH=""
ARG CJK_FONTS=0

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/models/hf \
    STORAGE_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 fontconfig \
    && if [ "$CJK_FONTS" = "1" ]; then apt-get install -y --no-install-recommends fonts-noto-cjk; fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

# CPU-only torch first (the PyPI wheel pulls ~3 GB of CUDA libraries); the project install
# below then finds `torch` already satisfied.
RUN case ",$EXTRAS," in *,torch,*) \
      uv pip install torch --index-url https://download.pytorch.org/whl/cpu ;; esac

WORKDIR /app
COPY . .
# editable: the package resolves weights/ and fonts/ next to itself, i.e. /app/weights and
# /app/fonts (a regular install would look for them inside site-packages)
RUN uv pip install -e ".[${EXTRAS}]"

RUN terjemahin-worker fetch-weights ${FETCH} && terjemahin-worker methods

EXPOSE 8000
CMD ["terjemahin-worker", "serve"]
