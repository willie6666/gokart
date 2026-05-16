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
    min_ocr_confidence: float
    ocr_mode: str
    ocr_max_side: int
    ocr_det_limit_side_len: int
    ocr_det_model: str
    ocr_rec_model: str
    ocr_cpu_threads: int
    ocr_rectify_table: bool
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
        min_ocr_confidence=float(os.environ.get("GOKART_MIN_OCR_CONFIDENCE", "0.50")),
        ocr_mode=os.environ.get("GOKART_OCR_MODE", "grid").strip().lower(),
        ocr_max_side=int(os.environ.get("GOKART_OCR_MAX_SIDE", "1600")),
        ocr_det_limit_side_len=int(os.environ.get("GOKART_OCR_DET_LIMIT_SIDE_LEN", "1600")),
        ocr_det_model=os.environ.get("GOKART_OCR_DET_MODEL", "PP-OCRv5_server_det"),
        ocr_rec_model=os.environ.get("GOKART_OCR_REC_MODEL", "PP-OCRv5_server_rec"),
        ocr_cpu_threads=int(os.environ.get("GOKART_OCR_CPU_THREADS", "1")),
        ocr_rectify_table=os.environ.get("GOKART_OCR_RECTIFY_TABLE", "false").strip().lower() not in {"0", "false", "no"},
        debug_ocr=os.environ.get("GOKART_DEBUG_OCR", "true").strip().lower() not in {"0", "false", "no"},
        debug_dir=Path(os.environ.get("GOKART_DEBUG_DIR", "data/debug")),
    )
