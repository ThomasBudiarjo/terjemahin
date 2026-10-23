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
    translator: str = "nllb"
    mask_refiner: str = "default"
    inpainter: str = "lama_manga"
    renderer: str = "default"
    source_lang: str = "ja"
    target_lang: str = "id"

    # weights
    weights_dir: Path = WORKER_ROOT / "weights"
    yolo_weights: Path = WORKER_ROOT / "weights" / "manga-text-yolo11n.onnx"
    ctd_weights: Path = WORKER_ROOT / "weights" / "comictextdetector-blk.onnx"
    detect_conf: float = 0.4
    detect_iou: float = 0.5
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
