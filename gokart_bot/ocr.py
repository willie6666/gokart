from __future__ import annotations

from pathlib import Path
import re

from .cell_ocr import CellOcrEngine
from .grid_parser import parse_grid_sheet, parse_grid_sheet_with_virtual_rows
from .lap_row_detector import detect_virtual_lap_rows
from .ocr_debug import write_cell_ocr, write_debug_parsed, write_ocr_words, write_paddleocr_overlay, write_parsed_overlay, write_virtual_rows_overlay
from .parser import ParsedSheet
from .table_grid import crop_column_region, extract_table_grid


class OcrEngine:
    def __init__(
        self,
        max_side: int = 1600,
        debug_dir: Path | str = Path("data"),
        ocr_lang: str = "ch",
        ocr_device: str = "cpu",
        ocr_enable_mkldnn: bool = False,
        ocr_cpu_threads: int = 1,
    ) -> None:
        self.max_side = max_side
        self.debug_dir = Path(debug_dir)
        self.cell_engine = CellOcrEngine(lang=ocr_lang, device=ocr_device, enable_mkldnn=ocr_enable_mkldnn, cpu_threads=ocr_cpu_threads)

    def recognize_lap_sheet(self, image_path: Path, session_id: str | None = None) -> ParsedSheet:
        debug_dir = self.debug_dir_for_session(session_id)
        grid = extract_table_grid(image_path, debug_dir, self.max_side)
        if grid.row_count < 8 or grid.col_count < 18:
            raise RuntimeError("無法辨識：無法偵測到完整表格")
        cell_engine = self.cell_engine
        table_words = cell_engine.recognize_image_words(grid.image)
        write_paddleocr_overlay(grid, table_words, debug_dir)
        header_cells = cell_engine.recognize_header_cells(grid, table_words)
        _add_full_top_header_cell(grid, header_cells, table_words)
        lap_cell = _find_lap_cell(grid, header_cells)
        if lap_cell is None:
            parsed = parse_grid_sheet(grid, header_cells)
            write_cell_ocr(header_cells, debug_dir)
            write_debug_parsed(parsed, debug_dir)
            write_parsed_overlay(grid, parsed, debug_dir)
            return parsed

        lap_row, lap_col = lap_cell
        lap_box = grid.cell(lap_row, lap_col)
        y_start = lap_box.y + lap_box.h if lap_box else grid.y_lines[min(lap_row + 1, len(grid.y_lines) - 1)]
        y_end = _find_avg_y(grid, header_cells, lap_row + 1)
        column_words = {}
        for col in range(lap_col, grid.col_count):
            region = crop_column_region(grid, col, y_start=y_start, y_end=int(y_end) if y_end else None)
            mode = "lap_index" if col == lap_col else "lap_time"
            column_words[col] = _column_words_from_table_words(table_words, region, mode)

        all_words = [word for words in column_words.values() for word in words]
        virtual_rows = detect_virtual_lap_rows(all_words, float(y_start), int(y_end or grid.image.shape[0]))
        parsed = parse_grid_sheet_with_virtual_rows(grid, header_cells, column_words)
        write_cell_ocr(header_cells, debug_dir)
        write_ocr_words(column_words, virtual_rows, debug_dir)
        write_virtual_rows_overlay(grid, column_words, virtual_rows, debug_dir, y_end)
        write_debug_parsed(parsed, debug_dir)
        write_parsed_overlay(grid, parsed, debug_dir)
        return parsed

    def debug_dir_for_session(self, session_id: str | None) -> Path:
        return self.debug_dir / str(session_id or "manual")


def _find_lap_cell(grid, header_cells) -> tuple[int, int] | None:
    from .grid_parser import looks_like_lap_nr

    matches = [(row, col) for (row, col), cell in header_cells.items() if looks_like_lap_nr(cell.raw_text or cell.normalized_text)]
    return sorted(matches, key=lambda item: (item[0], item[1]))[0] if matches else None


def _add_full_top_header_cell(grid, header_cells, table_words) -> None:
    y2 = grid.y_lines[1] if len(grid.y_lines) > 1 else max(60, grid.image.shape[0] // 8)
    y2 = min(grid.image.shape[0], max(y2, int(grid.image.shape[0] * 0.08)))
    words = [word for word in table_words if word.center_y <= y2]
    text = " ".join(word.text for word in sorted(words, key=lambda item: (item.y, item.x)))
    if text:
        header_cells[(-1, 0)] = _cell_result(-1, 0, text)


def _cell_result(row: int, col: int, raw: str, confidence: float | None = None):
    from .cell_ocr import CellOcrResult

    return CellOcrResult(row=row, col=col, raw_text=raw, normalized_text=" ".join(raw.strip().split()), confidence=confidence)


def _find_avg_y(grid, header_cells, start_row: int) -> float | None:
    from .grid_parser import looks_like_avg

    rows = [row for (row, _), cell in header_cells.items() if row >= start_row and looks_like_avg(cell.raw_text or cell.normalized_text)]
    if not rows:
        return None
    cell = grid.cell(min(rows), 0)
    return float(cell.y) if cell else None


def _column_words_from_table_words(table_words, region, mode: str):
    from .cell_ocr import normalize_lap_text, parse_lap_time
    from .lap_row_detector import looks_like_lap_candidate

    words = []
    x1 = region.x_offset
    x2 = region.x_offset + region.image.shape[1]
    y1 = region.y_offset
    y2 = region.y_offset + region.image.shape[0]
    for word in table_words:
        if not (x1 <= word.center_x < x2 and y1 <= word.center_y < y2):
            continue
        text = word.text.strip()
        if mode == "lap_index":
            text = re.sub(r"[^0-9]", "", text)
            if not text or not text.isdigit() or not (1 <= int(text) <= 80):
                continue
        else:
            text = re.sub(r"[^0-9.,:]", "", text)
            if re.match(r"\d{3,}[.,:]", text):
                text = _lap_time_suffix_candidate(text) or text
            if not looks_like_lap_candidate(text):
                cleaned = _lap_time_suffix_candidate(text)
                if cleaned is None:
                    continue
                text = cleaned
            if parse_lap_time(normalize_lap_text(text)) is None:
                continue
        words.append(type(word)(text, word.confidence, word.x, word.y, word.w, word.h))
    return sorted(words, key=lambda item: (item.center_y, item.center_x))


def _lap_time_suffix_candidate(text: str) -> str | None:
    from .cell_ocr import normalize_lap_text
    from .lap_row_detector import looks_like_lap_candidate

    for start in range(1, min(3, len(text)) + 1):
        candidate = text[start:]
        if looks_like_lap_candidate(candidate):
            return candidate
    normalized = normalize_lap_text(text)
    return normalized if looks_like_lap_candidate(normalized) else None
