from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    discord_token: str
    data_path: Path
    image_dir: Path
    ocr_max_side: int
    ocr_workers: int
    debug_ocr: bool
    debug_dir: Path


def load_config() -> Config:
    load_dotenv()

    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required. Copy .env.example to .env and set it.")

    return Config(
        discord_token=token,
        data_path=Path(os.environ.get("GOKART_DATA_PATH", "data/gokart_records.json")),
        image_dir=Path(os.environ.get("GOKART_IMAGE_DIR", "data/images")),
        ocr_max_side=int(os.environ.get("GOKART_OCR_MAX_SIDE", "1600")),
        ocr_workers=max(1, int(os.environ.get("GOKART_OCR_WORKERS", "1"))),
        debug_ocr=os.environ.get("GOKART_DEBUG_OCR", "true").strip().lower() not in {"0", "false", "no"},
        debug_dir=Path(os.environ.get("GOKART_DEBUG_DIR", "data/debug")),
    )
