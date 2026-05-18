from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    discord_token: str
    data_path: Path
    ocr_max_side: int
    ocr_workers: int
    ocr_lang: str
    ocr_device: str
    ocr_enable_mkldnn: bool
    ocr_cpu_threads: int


def load_config() -> Config:
    load_dotenv()

    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required. Copy .env.example to .env and set it.")

    return Config(
        discord_token=token,
        data_path=Path(os.environ.get("GOKART_DATA_PATH", "data")),
        ocr_max_side=int(os.environ.get("GOKART_OCR_MAX_SIDE", "1600")),
        ocr_workers=max(1, int(os.environ.get("GOKART_OCR_WORKERS", "1"))),
        ocr_lang=os.environ.get("GOKART_OCR_LANG", "ch").strip() or "ch",
        ocr_device=os.environ.get("GOKART_OCR_DEVICE", "cpu").strip() or "cpu",
        ocr_enable_mkldnn=os.environ.get("GOKART_OCR_ENABLE_MKLDNN", "false").strip().lower() not in {"0", "false", "no"},
        ocr_cpu_threads=max(1, int(os.environ.get("GOKART_OCR_CPU_THREADS", str(os.cpu_count() or 1)))),
    )
