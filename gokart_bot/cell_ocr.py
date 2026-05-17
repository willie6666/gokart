from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import cv2
import numpy as np

from .image_preprocess import make_ocr_image
from .lap_row_detector import TesseractWord, looks_like_lap_candidate
from .table_grid import TableGrid


TESSERACT_DPI = "--dpi 300 -c user_defined_dpi=300"


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
        if not self.has_tesseract:
            return "", None
        ocr_image = _preprocess(image, mode)
        result = _tesseract_string(ocr_image, mode)
        return (result, None) if result else ("", None)

    def recognize_header_cells(self, grid: TableGrid, debug_dir: Path | None = None) -> dict[tuple[int, int], CellOcrResult]:
        return self._recognize_grid_with_tesseract(grid, debug_dir)

    def recognize_text_region(self, image: np.ndarray) -> str:
        if not self.has_tesseract:
            return ""
        ocr_image = _upscale(image, target_height=260)
        return _tesseract_string(ocr_image, "text") or ""

    def recognize_integer_candidates(self, image: np.ndarray) -> list[str]:
        if not self.has_tesseract:
            return []
        ocr_image = _preprocess(image, "integer")
        result = _tesseract_string(ocr_image, "integer")
        digits = re.sub(r"[^0-9]", "", result)
        return [digits] if digits else []

    def recognize_column_words(self, image: np.ndarray, mode: str = "lap_time") -> list[TesseractWord]:
        if not self.has_tesseract:
            raise RuntimeError("Tesseract is required for virtual-row column OCR")
        words: list[TesseractWord] = []
        for crop, y_offset in _column_passes(image):
            ocr_image, scale = _upscale_with_scale(crop)
            pass_words = _tesseract_data_words(ocr_image, mode)
            pass_words = _scale_words(pass_words, 1 / scale)
            pass_words = _offset_words(pass_words, y_offset)
            words.extend(pass_words)

        deduped: dict[tuple[str, int, int], TesseractWord] = {}
        for word in words:
            if mode == "lap_time" and not looks_like_lap_candidate(word.text):
                continue
            if mode == "lap_index" and not _looks_like_lap_index(word.text):
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
    if value < 15.0 or value > 90.0:
        return None
    return value


def _preprocess(image: np.ndarray, mode: str) -> np.ndarray:
    """Single-pass preprocessing: clean edges, enhance, upscale."""
    target_height = {"integer": 180, "lap_time": 140, "text": 120}.get(mode, 140)
    if mode == "text":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        ocr = make_ocr_image(gray)
        return _resize_min(ocr, target_height)
    cleaned = _remove_cell_edges(image)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY) if cleaned.ndim == 3 else cleaned.copy()
    ocr = make_ocr_image(gray)
    return _resize_min(ocr, target_height)


def _tesseract_string(image: np.ndarray, mode: str) -> str:
    import pytesseract

    config = _tesseract_config(mode)
    try:
        return pytesseract.image_to_string(image, config=config).strip()
    except Exception:
        return ""


def _tesseract_data_words(image: np.ndarray, mode: str) -> list[TesseractWord]:
    import pytesseract
    from pytesseract import Output

    config = _tesseract_config(mode)
    try:
        data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
    except Exception:
        return []
    words: list[TesseractWord] = []
    for index, text in enumerate(data.get("text", [])):
        text = str(text).strip()
        if not text:
            continue
        try:
            confidence_value = float(data["conf"][index])
        except (ValueError, TypeError):
            confidence_value = -1.0
        words.append(
            TesseractWord(
                text=text,
                confidence=confidence_value if confidence_value >= 0 else None,
                x=float(data["left"][index]),
                y=float(data["top"][index]),
                w=float(data["width"][index]),
                h=float(data["height"][index]),
            )
        )
    return words


def _tesseract_config(mode: str) -> str:
    whitelist = {
        "integer": "0123456789",
        "lap_index": "0123456789",
        "lap_time": "0123456789.,:Il|Oo",
        "text": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/:. ",
    }.get(mode, "")
    psm = 10 if mode == "integer" else 6
    parts = [TESSERACT_DPI, f"--oem 1 --psm {psm}"]
    if whitelist:
        parts.append(f"-c tessedit_char_whitelist={whitelist}")
    return " ".join(parts)


def _upscale(image: np.ndarray, target_height: int = 260) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    ocr = make_ocr_image(gray)
    return _resize_min(ocr, target_height)


def _upscale_with_scale(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    ocr = make_ocr_image(gray)
    height = ocr.shape[0]
    if height < 900:
        scale = 900 / max(height, 1)
        new_w = max(1, int(ocr.shape[1] * scale))
        return cv2.resize(ocr, (new_w, 900), interpolation=cv2.INTER_CUBIC), scale
    return ocr, 1.0


def _resize_min(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height >= target_height:
        return image
    scale = min(max(target_height / max(height, 1), 2.0), 4.0)
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_CUBIC)


def _remove_cell_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.size == 0:
        return gray
    result = gray.copy()
    height, width = result.shape[:2]
    binary = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    edge_margin_x = max(2, int(width * 0.10))
    edge_margin_y = max(2, int(height * 0.10))
    for x in range(edge_margin_x):
        if np.count_nonzero(binary[:, x]) / max(height, 1) >= 0.70:
            result[:, x] = 255
    for x in range(max(0, width - edge_margin_x), width):
        if np.count_nonzero(binary[:, x]) / max(height, 1) >= 0.70:
            result[:, x] = 255
    for y in range(edge_margin_y):
        if np.count_nonzero(binary[y, :]) / max(width, 1) >= 0.70:
            result[y, :] = 255
    for y in range(max(0, height - edge_margin_y), height):
        if np.count_nonzero(binary[y, :]) / max(width, 1) >= 0.70:
            result[y, :] = 255
    return result


def _column_passes(image: np.ndarray) -> list[tuple[np.ndarray, float]]:
    height = image.shape[0]
    if height < 160:
        return [(image, 0.0)]
    midpoint = height // 2
    overlap = min(max(height // 12, 24), 80)
    return [
        (image, 0.0),
        (image[: min(height, midpoint + overlap)], 0.0),
        (image[max(0, midpoint - overlap) :], float(max(0, midpoint - overlap))),
    ]


def _scale_words(words: list[TesseractWord], scale: float) -> list[TesseractWord]:
    if scale == 1.0:
        return words
    return [
        TesseractWord(text=w.text, confidence=w.confidence, x=w.x * scale, y=w.y * scale, w=w.w * scale, h=w.h * scale)
        for w in words
    ]


def _offset_words(words: list[TesseractWord], y_offset: float) -> list[TesseractWord]:
    if y_offset == 0:
        return words
    return [
        TesseractWord(text=w.text, confidence=w.confidence, x=w.x, y=w.y + y_offset, w=w.w, h=w.h)
        for w in words
    ]


def _looks_like_lap_index(text: str) -> bool:
    stripped = re.sub(r"[^0-9]", "", text)
    if not stripped:
        return False
    value = int(stripped)
    return 1 <= value <= 80


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
