from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

WORKER_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # queue
    amqp_url: str = "amqp://guest:guest@localhost:5672/"
    amqp_queue: str = "terjemahin.jobs"
    storage_dir: Path = Path("./data")

    # llm
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    # detector
    detector: Literal["yolo", "ctd"] = "yolo"
    yolo_weights: Path = WORKER_ROOT / "weights" / "manga-text-yolo11n.onnx"
    ctd_weights: Path = WORKER_ROOT / "weights" / "comictextdetector.pt.onnx"
    detect_conf: float = 0.4
    detect_iou: float = 0.5

    # typeset
    font_path: Path = WORKER_ROOT / "fonts" / "default.ttf"


def get_settings() -> Settings:
    return Settings()
