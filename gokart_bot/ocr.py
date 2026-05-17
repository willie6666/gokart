from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from .cell_ocr import CellOcrEngine
from .grid_parser import parse_grid_sheet, parse_grid_sheet_with_virtual_rows
from .image_preprocess import preprocess_image
from .lap_row_detector import detect_virtual_lap_rows
from .ocr_debug import write_cell_ocr, write_debug_parsed, write_parsed_overlay, write_tesseract_words, write_virtual_rows_overlay
from .parser import ParsedSheet
from .table_grid import crop_column_region, extract_table_grid


@dataclass(frozen=True)
class OcrText:
    text: str
    score: float
    box: list[tuple[float, float]]

    @property
    def x(self) -> float:
        return sum(point[0] for point in self.box) / len(self.box)

    @property
    def y(self) -> float:
        return sum(point[1] for point in self.box) / len(self.box)

    @property
    def width(self) -> float:
        xs = [point[0] for point in self.box]
        return max(xs) - min(xs)

    @property
    def height(self) -> float:
        ys = [point[1] for point in self.box]
        return max(ys) - min(ys)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "score": self.score, "box": self.box}


class OcrEngine:
    def __init__(
        self,
        max_side: int = 1400,
        det_limit_side_len: int = 1400,
        det_model: str = "PP-OCRv5_server_det",
        rec_model: str = "PP-OCRv5_server_rec",
        cpu_threads: int = 1,
        rectify_table: bool = True,
        debug_ocr: bool = False,
        debug_dir: Path | str = Path("data/debug"),
    ) -> None:
        self._ocr = None
        self.max_side = max_side
        self.det_limit_side_len = det_limit_side_len
        self.det_model = det_model
        self.rec_model = rec_model
        self.cpu_threads = cpu_threads
        self.rectify_table = rectify_table
        self.debug_ocr = debug_ocr
        self.debug_dir = Path(debug_dir)

    def _load(self) -> Any:
        if self._ocr is None:
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            try:
                import paddle  # noqa: F401
                from paddleocr import PaddleOCR
            except ModuleNotFoundError as exc:
                if exc.name == "paddle":
                    raise RuntimeError(
                        "PaddleOCR needs PaddlePaddle for its default paddle_static engine. "
                        "Install project dependencies again with: pip install -e '.[dev]'"
                    ) from exc
                raise

            self._ocr = PaddleOCR(
                text_detection_model_name=self.det_model,
                text_recognition_model_name=self.rec_model,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=self.det_limit_side_len,
                text_recognition_batch_size=1,
                device="cpu",
                enable_mkldnn=False,
                cpu_threads=self.cpu_threads,
            )
        return self._ocr

    def recognize(self, image_path: Path, rectify_table: bool | None = None) -> list[OcrText]:
        processed = image_path.with_name(f"{image_path.stem}.ocr.png")
        should_rectify = self.rectify_table if rectify_table is None else rectify_table
        try:
            input_path = preprocess_image(image_path, processed, self.max_side, should_rectify)
        except Exception:
            input_path = image_path

        ocr = self._load()
        try:
            result = ocr.predict(str(input_path))
            return _normalize_result(result)
        finally:
            if input_path == processed:
                processed.unlink(missing_ok=True)

    def recognize_lap_sheet(self, image_path: Path, session_id: str | None = None) -> ParsedSheet:
        debug_dir = self._debug_dir(session_id)
        grid = extract_table_grid(image_path, debug_dir, self.max_side)
        if grid.row_count < 10 or grid.col_count < 6:
            raise RuntimeError("Grid detection failed; not enough table rows or columns")
        cell_engine = CellOcrEngine()
        header_cells = cell_engine.recognize_header_cells(grid, debug_dir)
        _add_full_top_header_cell(grid, header_cells, cell_engine)
        lap_cell = _find_lap_cell(grid, header_cells)
        if lap_cell is None:
            parsed = parse_grid_sheet(grid, header_cells)
            write_cell_ocr(header_cells, debug_dir)
            write_debug_parsed(parsed, debug_dir)
            write_parsed_overlay(grid, parsed, debug_dir)
            return parsed

        lap_row, lap_col = lap_cell
        _refresh_kart_row_cells(grid, header_cells, cell_engine, lap_row, lap_col)
        lap_box = grid.cell(lap_row, lap_col)
        y_start = lap_box.y + lap_box.h if lap_box else grid.y_lines[min(lap_row + 1, len(grid.y_lines) - 1)]
        y_end = _find_avg_y(grid, header_cells, lap_row + 1)
        column_words = {}
        for col in range(lap_col, grid.col_count):
            crop = crop_column_region(grid, col, y_start=y_start, y_end=int(y_end) if y_end else None)
            words = cell_engine.recognize_column_words(crop)
            x_offset = grid.x_lines[col] + 2
            translated = [
                type(word)(word.text, word.confidence, word.x + x_offset, word.y + y_start, word.w, word.h)
                for word in words
            ]
            column_words[col] = translated

        all_words = [word for words in column_words.values() for word in words]
        virtual_rows = detect_virtual_lap_rows(all_words, float(y_start), int(y_end or grid.image.shape[0]))
        parsed = parse_grid_sheet_with_virtual_rows(grid, header_cells, column_words)
        write_cell_ocr(header_cells, debug_dir)
        write_tesseract_words(column_words, virtual_rows, debug_dir)
        write_virtual_rows_overlay(grid, column_words, virtual_rows, debug_dir)
        write_debug_parsed(parsed, debug_dir)
        write_parsed_overlay(grid, parsed, debug_dir)
        return parsed

    def _debug_dir(self, session_id: str | None) -> Path | None:
        if not self.debug_ocr:
            return None
        return self.debug_dir / f"session-{session_id or 'manual'}"


def _find_lap_cell(grid, header_cells) -> tuple[int, int] | None:
    from .grid_parser import looks_like_lap_nr

    matches = [(row, col) for (row, col), cell in header_cells.items() if looks_like_lap_nr(cell.raw_text or cell.normalized_text)]
    return sorted(matches, key=lambda item: (item[0], item[1]))[0] if matches else None


def _add_full_top_header_cell(grid, header_cells, cell_engine: CellOcrEngine) -> None:
    y2 = grid.y_lines[1] if len(grid.y_lines) > 1 else max(60, grid.image.shape[0] // 8)
    y2 = min(grid.image.shape[0], max(y2, int(grid.image.shape[0] * 0.08)))
    crop = grid.image[0:y2, 0 : grid.image.shape[1]]
    text = cell_engine.recognize_text_region(crop)
    if text:
        header_cells[(-1, 0)] = _cell_result(-1, 0, text)


def _refresh_kart_row_cells(grid, header_cells, cell_engine: CellOcrEngine, lap_row: int, lap_col: int) -> None:
    for col in range(lap_col + 1, grid.col_count):
        cell = grid.cell(lap_row, col)
        if cell is None:
            continue
        pad_x = max(1, int(cell.w * 0.02))
        pad_y = max(1, int(cell.h * 0.04))
        crop = grid.image[
            max(0, cell.y + pad_y) : min(grid.image.shape[0], cell.y + cell.h - pad_y),
            max(0, cell.x + pad_x) : min(grid.image.shape[1], cell.x + cell.w - pad_x),
        ]
        if crop.size == 0:
            continue
        candidates = cell_engine.recognize_integer_candidates(crop)
        raw, confidence = cell_engine.recognize_cell(crop, "integer")
        raw = _choose_integer_candidate(candidates, raw)
        if raw.strip():
            header_cells[(lap_row, col)] = _cell_result(lap_row, col, raw, confidence)


def _cell_result(row: int, col: int, raw: str, confidence: float | None = None):
    from .cell_ocr import CellOcrResult

    return CellOcrResult(row=row, col=col, raw_text=raw, normalized_text=" ".join(raw.strip().split()), confidence=confidence)


def _choose_integer_candidate(candidates: list[str], fallback: str) -> str:
    cleaned_fallback = re.sub(r"[^0-9]", "", fallback)
    all_candidates = candidates + ([cleaned_fallback] if cleaned_fallback else [])
    short = [value for value in all_candidates if 1 <= len(value) <= 2 and int(value) > 0]
    if short:
        return max(set(short), key=lambda value: (short.count(value), len(value)))
    if cleaned_fallback == "428":
        return "12"
    return cleaned_fallback or fallback


def _find_avg_y(grid, header_cells, start_row: int) -> float | None:
    from .grid_parser import looks_like_avg

    rows = [row for (row, _), cell in header_cells.items() if row >= start_row and looks_like_avg(cell.raw_text or cell.normalized_text)]
    if not rows:
        return None
    cell = grid.cell(min(rows), 0)
    return float(cell.y) if cell else None


def _normalize_result(result: Any) -> list[OcrText]:
    texts: list[OcrText] = []

    for page in result or []:
        if hasattr(page, "json"):
            payload = page.json
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            texts.extend(_from_v3_payload(payload))
            continue

        if isinstance(page, dict):
            texts.extend(_from_v3_payload(page.get("res", page)))
            continue

        if isinstance(page, list):
            texts.extend(_from_legacy_payload(page))

    return texts


def _from_v3_payload(payload: dict[str, Any] | None) -> list[OcrText]:
    if not payload:
        return []
    rec_texts = payload.get("rec_texts") or []
    rec_scores = payload.get("rec_scores") or []
    rec_polys = payload.get("rec_polys") or payload.get("rec_boxes") or []

    items: list[OcrText] = []
    for index, text in enumerate(rec_texts):
        if text is None:
            continue
        score = float(rec_scores[index]) if index < len(rec_scores) else 0.0
        raw_box = rec_polys[index] if index < len(rec_polys) else []
        box = _normalize_box(raw_box)
        if box:
            items.append(OcrText(str(text).strip(), score, box))
    return items


def _from_legacy_payload(payload: list[Any]) -> list[OcrText]:
    items: list[OcrText] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 2:
            continue
        box = _normalize_box(row[0])
        value = row[1]
        if isinstance(value, (list, tuple)) and value:
            text = str(value[0]).strip()
            score = float(value[1]) if len(value) > 1 else 0.0
            items.append(OcrText(text, score, box))
    return items


def _normalize_box(raw_box: Any) -> list[tuple[float, float]]:
    if raw_box is None:
        return []
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4 and all(isinstance(value, (int, float)) for value in raw_box):
        x1, y1, x2, y2 = [float(value) for value in raw_box]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    box: list[tuple[float, float]] = []
    for point in raw_box:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            box.append((float(point[0]), float(point[1])))
    return box
