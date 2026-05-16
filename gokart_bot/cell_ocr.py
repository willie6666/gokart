from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image_preprocess import make_ocr_image
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
    def __init__(self, paddle_ocr=None) -> None:
        self.paddle_ocr = paddle_ocr
        self.has_tesseract = _has_tesseract()

    def recognize_cell(self, image: np.ndarray, mode: str) -> tuple[str, float | None]:
        ocr_image = make_ocr_image(image)
        height, width = ocr_image.shape[:2]
        max_side = max(width, height)
        if max_side > 320:
            scale = 320 / max_side
            ocr_image = cv2.resize(ocr_image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        tesseract = _try_tesseract(ocr_image, mode) if self.has_tesseract else None
        if tesseract is not None:
            return tesseract, None
        if self.paddle_ocr is None:
            return "", None
        paddle_image = cv2.cvtColor(ocr_image, cv2.COLOR_GRAY2BGR) if ocr_image.ndim == 2 else ocr_image
        paddle_image = _pad_for_paddle_detector(paddle_image)
        result = self.paddle_ocr.predict(paddle_image)
        texts: list[str] = []
        scores: list[float] = []
        for page in result or []:
            payload = page.json if hasattr(page, "json") else page
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            if isinstance(payload, dict):
                texts.extend(str(text) for text in payload.get("rec_texts", []) if text is not None)
                scores.extend(float(score) for score in payload.get("rec_scores", []) if score is not None)
        return " ".join(texts).strip(), round(sum(scores) / len(scores), 3) if scores else None

    def recognize_grid(self, grid: TableGrid, debug_dir: Path | None = None) -> dict[tuple[int, int], CellOcrResult]:
        if self.paddle_ocr is not None:
            paddle_results = self._recognize_grid_with_paddle_table(grid, debug_dir)
            if self.has_tesseract:
                return _merge_tesseract_headers_with_paddle_laps(self._recognize_grid_with_tesseract(grid, debug_dir), paddle_results)
            return paddle_results

        return self._recognize_grid_with_tesseract(grid, debug_dir)

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

    def _recognize_grid_with_paddle_table(
        self, grid: TableGrid, debug_dir: Path | None = None
    ) -> dict[tuple[int, int], CellOcrResult]:
        image = grid.image
        max_side = max(image.shape[:2])
        if max_side > 1800:
            scale = 1800 / max_side
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        result = self.paddle_ocr.predict(image)
        items = _paddle_items(result)
        if debug_dir:
            _write_paddle_items(items, debug_dir)
        grouped: dict[tuple[int, int], list[tuple[str, float | None, float]]] = {}
        for text, confidence, x, y in items:
            row_col = _cell_at(grid, x / scale, y / scale)
            if row_col is None:
                continue
            grouped.setdefault(row_col, []).append((text, confidence, x))

        results: dict[tuple[int, int], CellOcrResult] = {}
        for cell in grid.cells:
            values = sorted(grouped.get((cell.row, cell.col), []), key=lambda item: item[2])
            raw = " ".join(value[0] for value in values).strip()
            scores = [value[1] for value in values if value[1] is not None]
            confidence = round(sum(scores) / len(scores), 3) if scores else None
            mode = _mode_for_cell(cell.row, cell.col)
            normalized = normalize_lap_text(raw) if mode == "lap_time" else _normalize_text(raw)
            results[(cell.row, cell.col)] = CellOcrResult(cell.row, cell.col, raw, normalized, confidence)
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
    config = "--psm 7"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_string(image, config=config).strip()
    except Exception:
        return None


def _has_tesseract() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def _pad_for_paddle_detector(image: np.ndarray, max_aspect_ratio: float = 2.5) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width / height <= max_aspect_ratio:
        return image
    target_height = int(round(width / max_aspect_ratio))
    pad_total = max(0, target_height - height)
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    return cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _paddle_items(result) -> list[tuple[str, float | None, float, float]]:
    items: list[tuple[str, float | None, float, float]] = []
    for page in result or []:
        payload = page.json if hasattr(page, "json") else page
        if isinstance(payload, dict) and "res" in payload:
            payload = payload["res"]
        if not isinstance(payload, dict):
            continue
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        boxes = payload.get("rec_polys") or payload.get("rec_boxes") or []
        for index, text in enumerate(texts):
            if text is None or index >= len(boxes):
                continue
            box = np.asarray(boxes[index], dtype="float32")
            if box.ndim == 1 and len(box) == 4:
                x1, y1, x2, y2 = box.tolist()
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
            elif box.ndim == 2 and box.shape[1] >= 2:
                x = float(box[:, 0].mean())
                y = float(box[:, 1].mean())
            else:
                continue
            confidence = float(scores[index]) if index < len(scores) else None
            items.append((str(text).strip(), confidence, x, y))
    return items


def _write_paddle_items(items: list[tuple[str, float | None, float, float]], debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {"text": text, "confidence": confidence, "x": round(x, 2), "y": round(y, 2)}
        for text, confidence, x, y in items
    ]
    with (debug_dir / "paddle_ocr_items.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _merge_tesseract_headers_with_paddle_laps(
    tesseract_results: dict[tuple[int, int], CellOcrResult],
    paddle_results: dict[tuple[int, int], CellOcrResult],
) -> dict[tuple[int, int], CellOcrResult]:
    merged = dict(tesseract_results)
    lap_row = _find_lap_row(tesseract_results)
    for key, paddle_cell in paddle_results.items():
        if not paddle_cell.raw_text.strip():
            continue
        tesseract_cell = merged.get(key)
        if tesseract_cell is None or not tesseract_cell.raw_text.strip() or (lap_row is not None and key[0] > lap_row):
            merged[key] = paddle_cell
    return merged


def _find_lap_row(cells: dict[tuple[int, int], CellOcrResult]) -> int | None:
    for (row, _), cell in sorted(cells.items()):
        normalized = re.sub(r"[^a-z]", "", cell.raw_text.lower())
        if normalized in {"lapnr", "lpnr", "lapn", "lpn", "lapno", "lapnumber", "lap"}:
            return row
    return None


def _cell_at(grid: TableGrid, x: float, y: float) -> tuple[int, int] | None:
    row = _line_interval(grid.y_lines, y)
    col = _line_interval(grid.x_lines, x)
    if row is None or col is None:
        return None
    return row, col


def _line_interval(lines: list[int], value: float) -> int | None:
    for index in range(len(lines) - 1):
        if lines[index] <= value < lines[index + 1]:
            return index
    return None


def _mode_for_cell(row: int, col: int) -> str:
    if row <= 2 or col <= 1:
        return "text"
    if row <= 4:
        return "integer"
    return "lap_time"


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())
