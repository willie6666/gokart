from __future__ import annotations

from dataclasses import dataclass
import os
import re
import threading

import cv2
import numpy as np

from .lap_row_detector import OcrWord
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
    def __init__(self, lang: str = "ch", device: str = "cpu", enable_mkldnn: bool = True, cpu_threads: int = 1) -> None:
        self._local = threading.local()
        self.lang = lang
        self.device = device
        self.enable_mkldnn = enable_mkldnn
        self.cpu_threads = cpu_threads

    def recognize_image_words(self, image: np.ndarray) -> list[OcrWord]:
        return self._predict_words(image)

    def recognize_header_cells(
        self,
        grid: TableGrid,
        words: list[OcrWord] | None = None,
    ) -> dict[tuple[int, int], CellOcrResult]:
        return self._recognize_grid(grid, words)

    def _get_paddleocr(self):
        if self.enable_mkldnn:
            return self._create_paddleocr()
        reader = getattr(self._local, "paddleocr_reader", None)
        if reader is None:
            reader = self._create_paddleocr()
            self._local.paddleocr_reader = reader
        return reader

    def _create_paddleocr(self):
        os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        return PaddleOCR(
            lang=self.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self.device,
            enable_mkldnn=self.enable_mkldnn,
            cpu_threads=self.cpu_threads,
        )

    def _predict_words(self, image: np.ndarray) -> list[OcrWord]:
        if image.size == 0:
            return []
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result = self._get_paddleocr().predict(image)
        words: list[OcrWord] = []
        for page in result:
            payload = page.json.get("res", {}) if hasattr(page, "json") else {}
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            boxes = payload.get("rec_boxes") or []
            polys = payload.get("rec_polys") or []
            for index, text in enumerate(texts):
                box = boxes[index] if index < len(boxes) else _poly_to_box(polys[index] if index < len(polys) else None)
                if box is None:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                words.append(
                    OcrWord(
                        text=str(text).strip(),
                        confidence=float(scores[index]) if index < len(scores) else None,
                        x=x1,
                        y=y1,
                        w=max(0.0, x2 - x1),
                        h=max(0.0, y2 - y1),
                    )
                )
        return sorted(words, key=lambda item: (item.y, item.x))

    def _recognize_grid(
        self,
        grid: TableGrid,
        words: list[OcrWord] | None = None,
    ) -> dict[tuple[int, int], CellOcrResult]:
        results: dict[tuple[int, int], CellOcrResult] = {}
        grouped: dict[tuple[int, int], list[OcrWord]] = {}
        for word in words if words is not None else self.recognize_image_words(grid.image):
            row = _line_index(grid.y_lines, word.center_y)
            col = _line_index(grid.x_lines, word.center_x)
            if row is None or col is None:
                continue
            grouped.setdefault((row, col), []).append(word)
        for cell in grid.cells:
            mode = _mode_for_cell(cell.row, cell.col)
            items = sorted(grouped.get((cell.row, cell.col), []), key=lambda item: (item.y, item.x))
            raw = " ".join(item.text for item in items)
            raw = _postprocess_ocr_text(raw, mode)
            scores = [item.confidence for item in items if item.confidence is not None]
            confidence = (sum(scores) / len(scores)) if scores else None
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
    if not re.fullmatch(r"\d{1,2}\.\d{2,3}", text):
        digits = re.sub(r"[^0-9]", "", text)
        if re.fullmatch(r"\d{4}", digits):
            return f"{digits[:2]}.{digits[2:]}"
        if re.fullmatch(r"\d{5}", digits):
            return f"{digits[:2]}.{digits[2:]}"
    if re.fullmatch(r"\d{4}", text):
        text = f"{text[:2]}.{text[2:]}"
    elif re.fullmatch(r"\d{5}", text):
        text = f"{text[:2]}.{text[2:]}"
    elif "." not in text and len(text) >= 3:
        text = text[:-2] + "." + text[-2:]
    if "." not in text:
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 3:
            text = digits[:-2] + "." + digits[-2:]
    return text


def parse_lap_time(text: str) -> float | None:
    normalized = normalize_lap_text(text)
    if not re.fullmatch(r"\d{1,2}\.\d{2,3}", normalized):
        return None
    value = round(float(normalized), 3)
    if value < 15.0 or value > 90.0:
        return None
    return value


def _mode_for_cell(row: int, col: int) -> str:
    if row <= 2 or col <= 1:
        return "text"
    if row <= 4:
        return "integer"
    return "lap_time"


def _postprocess_ocr_text(text: str, mode: str) -> str:
    text = " ".join(text.strip().split())
    if mode in {"integer", "lap_index"}:
        return re.sub(r"[^0-9]", "", text)
    if mode == "lap_time":
        return re.sub(r"[^0-9.,:]", "", text)
    return re.sub(r"[^0-9A-Za-z上午下午/:. ：]", "", text)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _poly_to_box(poly) -> list[float] | None:
    if not poly:
        return None
    xs = [float(point[0]) for point in poly]
    ys = [float(point[1]) for point in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _line_index(lines: list[int], value: float) -> int | None:
    for index in range(len(lines) - 1):
        if lines[index] <= value < lines[index + 1]:
            return index
    return None
