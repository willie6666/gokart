from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import cv2
import numpy as np

from .image_preprocess import make_ocr_image
from .lap_row_detector import TesseractWord, looks_like_lap_candidate
from .table_grid import TableGrid


@dataclass(frozen=True)
class CellOcrResult:
    row: int
    col: int
    raw_text: str
    normalized_text: str
    confidence: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "row": self.row,
            "col": self.col,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
        }


class CellOcrEngine:
    def __init__(self) -> None:
        self.has_tesseract = _has_tesseract()

    def recognize_cell(self, image: np.ndarray, mode: str) -> tuple[str, float | None]:
        ocr_image = make_ocr_image(image)
        height, width = ocr_image.shape[:2]
        if height < 48:
            scale = 48 / max(height, 1)
            ocr_image = cv2.resize(ocr_image, (max(1, int(width * scale)), 48), interpolation=cv2.INTER_CUBIC)
        tesseract = _try_tesseract(ocr_image, mode) if self.has_tesseract else None
        if tesseract is not None:
            return tesseract, None
        return "", None

    def recognize_header_cells(self, grid: TableGrid, debug_dir: Path | None = None) -> dict[tuple[int, int], CellOcrResult]:
        return self._recognize_grid_with_tesseract(grid, debug_dir)

    def recognize_text_region(self, image: np.ndarray) -> str:
        if not self.has_tesseract:
            return ""
        try:
            import pytesseract
        except ModuleNotFoundError:
            return ""
        variants = make_tesseract_variants(upscale_header_for_tesseract(image))
        outputs: list[str] = []
        for variant in variants:
            for config in ("--oem 1 --psm 6", "--oem 1 --psm 11", "--oem 1 --psm 7"):
                try:
                    text = pytesseract.image_to_string(variant, config=config).strip()
                except Exception:
                    continue
                if text:
                    outputs.append(" ".join(text.split()))
        return max(outputs, key=len) if outputs else ""

    def recognize_integer_candidates(self, image: np.ndarray) -> list[str]:
        if not self.has_tesseract:
            return []
        try:
            import pytesseract
        except ModuleNotFoundError:
            return []
        candidates: list[str] = []
        variants = make_tesseract_variants(upscale_header_for_tesseract(image, target_height=180, max_aspect_ratio=4.0))
        for variant in variants:
            for config in (
                "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
                "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789",
                "--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789",
                "--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789",
            ):
                try:
                    text = pytesseract.image_to_string(variant, config=config).strip()
                except Exception:
                    continue
                digits = re.sub(r"[^0-9]", "", text)
                if digits:
                    candidates.append(digits)
        return candidates

    def recognize_column_words(self, image: np.ndarray, mode: str = "lap_time") -> list[TesseractWord]:
        if not self.has_tesseract:
            raise RuntimeError("Tesseract is required for virtual-row column OCR")
        words: list[TesseractWord] = []
        ocr_image, scale = upscale_for_tesseract(image)
        for variant in make_tesseract_variants(ocr_image):
            words.extend(_scale_words(_tesseract_data_words(variant, mode), 1 / scale))

        deduped: dict[tuple[str, int, int], TesseractWord] = {}
        for word in words:
            if mode == "lap_time" and not looks_like_lap_candidate(word.text):
                continue
            key = (normalize_lap_text(word.text) if mode == "lap_time" else word.text.strip(), round(word.center_x / 4), round(word.center_y / 4))
            current = deduped.get(key)
            if current is None or (word.confidence or 0) > (current.confidence or 0):
                deduped[key] = word
        return sorted(deduped.values(), key=lambda item: (item.center_y, item.center_x))

    def _recognize_grid_with_tesseract(self, grid: TableGrid, debug_dir: Path | None = None) -> dict[tuple[int, int], CellOcrResult]:
        cells_dir = debug_dir / "cells" if debug_dir else None
        if cells_dir:
            cells_dir.mkdir(parents=True, exist_ok=True)
        results: dict[tuple[int, int], CellOcrResult] = {}
        for cell in grid.cells:
            pad_x = max(2, int(cell.w * 0.06))
            pad_y = max(2, int(cell.h * 0.10))
            crop = grid.image[cell.y + pad_y : cell.y + cell.h - pad_y, cell.x + pad_x : cell.x + cell.w - pad_x]
            if crop.size == 0:
                continue
            mode = _mode_for_cell(cell.row, cell.col)
            raw, confidence = self.recognize_cell(crop, mode)
            normalized = normalize_lap_text(raw) if mode == "lap_time" else _normalize_text(raw)
            results[(cell.row, cell.col)] = CellOcrResult(cell.row, cell.col, raw, normalized, confidence)
            if cells_dir:
                cv2.imwrite(str(cells_dir / f"r{cell.row:02d}_c{cell.col:02d}.png"), crop)
        return results


def normalize_kart_no(text: str) -> int | None:
    text = text.strip()
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1").replace("|", "1")
    if re.fullmatch(r"\d{4,}", text):
        return None
    tokens = re.findall(r"\d{1,3}", text)
    for token in tokens:
        value = int(token)
        if 0 < value <= 999:
            return value
    return None


def normalize_lap_text(text: str) -> str:
    text = text.strip()
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1").replace("|", "1")
    text = text.replace(",", ".").replace(":", ".")
    text = re.sub(r"[^0-9.]", "", text)
    if re.fullmatch(r"\d{4}", text):
        text = f"{text[:2]}.{text[2:]}"
    if re.fullmatch(r"\d{5}", text):
        text = f"{text[:2]}.{text[2:]}"
    return text


def parse_lap_time(text: str) -> float | None:
    normalized = normalize_lap_text(text)
    if not re.fullmatch(r"\d{1,2}\.\d{2,3}", normalized):
        return None
    value = round(float(normalized), 3)
    if value < 10.0 or value > 90.0:
        return None
    return value


def _try_tesseract(image: np.ndarray, mode: str) -> str | None:
    try:
        import pytesseract
    except ModuleNotFoundError:
        return None
    whitelist = {
        "integer": "0123456789",
        "lap_time": "0123456789.,:Il|Oo",
        "text": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/:. ",
    }.get(mode, "")
    config = "--oem 1 --psm 7"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_string(image, config=config).strip()
    except Exception:
        return None


def upscale_for_tesseract(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if height < 900:
        scale = 900 / max(height, 1)
        return cv2.resize(image, (max(1, int(width * scale)), 900), interpolation=cv2.INTER_CUBIC), scale
    return image, 1.0


def upscale_header_for_tesseract(image: np.ndarray, target_height: int = 260, max_aspect_ratio: float = 8.0) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0:
        return image
    scale = max(target_height / height, 1.0)
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_CUBIC)
    resized_height, resized_width = resized.shape[:2]
    target_padded_height = max(resized_height, int(resized_width / max_aspect_ratio))
    if target_padded_height <= resized_height:
        return resized
    pad_total = target_padded_height - resized_height
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _scale_words(words: list[TesseractWord], scale: float) -> list[TesseractWord]:
    if scale == 1.0:
        return words
    return [
        TesseractWord(
            text=word.text,
            confidence=word.confidence,
            x=word.x * scale,
            y=word.y * scale,
            w=word.w * scale,
            h=word.h * scale,
        )
        for word in words
    ]


def make_tesseract_variants(image: np.ndarray) -> list[np.ndarray]:
    base = make_ocr_image(image)
    _, otsu = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    return [base, otsu, adaptive]


def _tesseract_data_words(image: np.ndarray, mode: str) -> list[TesseractWord]:
    try:
        import pytesseract
        from pytesseract import Output
    except ModuleNotFoundError:
        return []
    whitelist = {
        "integer": "0123456789",
        "lap_time": "0123456789.,:Il|Oo",
        "text": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/:. ",
    }.get(mode, "")
    configs = ["--oem 1 --psm 6", "--oem 1 --psm 11"] if mode == "lap_time" else ["--oem 1 --psm 7"]
    if whitelist:
        configs = [f"{config} -c tessedit_char_whitelist={whitelist}" for config in configs]
    words: list[TesseractWord] = []
    for config in configs:
        try:
            data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
        except Exception:
            continue
        for index, text in enumerate(data.get("text", [])):
            text = str(text).strip()
            if not text:
                continue
            try:
                confidence_value = float(data["conf"][index])
            except (ValueError, TypeError):
                confidence_value = -1.0
            confidence = confidence_value if confidence_value >= 0 else None
            words.append(
                TesseractWord(
                    text=text,
                    confidence=confidence,
                    x=float(data["left"][index]),
                    y=float(data["top"][index]),
                    w=float(data["width"][index]),
                    h=float(data["height"][index]),
                )
            )
    return words


def _has_tesseract() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def _mode_for_cell(row: int, col: int) -> str:
    if row <= 2 or col <= 1:
        return "text"
    if row <= 4:
        return "integer"
    return "lap_time"


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())
