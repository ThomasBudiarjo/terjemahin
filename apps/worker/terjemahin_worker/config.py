from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

WORKER_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # queue
    amqp_url: str = "amqp://guest:guest@localhost:5672/"
    amqp_queue: str = "terjemahin.jobs"
    storage_dir: Path = Path("./data")

    # default method per stage (see `terjemahin-worker methods`)
    detector: str = "ctd"
    ocr: str = "manga_ocr"
    translator: str = "m2m100"  # best offline seq2seq on page01 + eval/fixtures/lines_ja_id.tsv
    mask_refiner: str = "default"
    inpainter: str = "telea"
    renderer: str = "default"
    source_lang: str = "ja"
    target_lang: str = "id"
    # OCR used when none is chosen and the source is not Japanese (manga_ocr reads Japanese only)
    ocr_non_japanese: str = "48px"

    # robustness limits
    max_pixels: int = 50_000_000  # larger pages are refused before decoding (memory)
    min_ocr_confidence: float = 0.4  # OCR reads below this are left untouched (hallucinations)
    stage_cache_size: int = 2  # built instances kept per stage (each may hold a model)
    web_keep_jobs: int = 100  # web UI jobs kept on disk (~13 MB each); 0 keeps all

    # weights
    weights_dir: Path = WORKER_ROOT / "weights"
    yolo_weights: Path = WORKER_ROOT / "weights" / "manga-text-yolo11n.onnx"
    ctd_weights: Path = WORKER_ROOT / "weights" / "comictextdetector-blk.onnx"
    detect_conf: float = 0.4
    detect_iou: float = 0.5
    # split boxes laid over two balloons, grow boxes that stop short of their text
    # (detect/postprocess.py); applies to every detector
    detect_refine: bool = True
    device: str = "cpu"  # torch-based methods: cpu | cuda | mps

    # llm / api translators
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    # typeset
    font_path: Path = WORKER_ROOT / "fonts" / "ComicShanns2.ttf"


def get_settings() -> Settings:
    return Settings()
