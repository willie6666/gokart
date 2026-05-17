from __future__ import annotations

from pathlib import Path

import cv2

from .cell_ocr import CellOcrEngine, preprocess_image
from .grid_parser import parse_grid_sheet, parse_grid_sheet_with_virtual_rows
from .lap_row_detector import detect_virtual_lap_rows
from .ocr_debug import write_cell_ocr, write_debug_parsed, write_parsed_overlay, write_tesseract_words, write_virtual_rows_overlay
from .parser import ParsedSheet
from .table_grid import crop_column_region, extract_table_grid


class OcrEngine:
    def __init__(
        self,
        max_side: int = 1600,
        debug_ocr: bool = False,
        debug_dir: Path | str = Path("data/debug"),
    ) -> None:
        self.max_side = max_side
        self.debug_ocr = debug_ocr
        self.debug_dir = Path(debug_dir)

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
        crops_dir = debug_dir / "column_crops" if debug_dir else None
        if crops_dir:
            crops_dir.mkdir(parents=True, exist_ok=True)
        for col in range(lap_col, grid.col_count):
            region = crop_column_region(grid, col, y_start=y_start, y_end=int(y_end) if y_end else None)
            mode = "lap_index" if col == lap_col else "lap_time"
            if crops_dir:
                label = "lap_index" if col == lap_col else f"col{col:02d}"
                cv2.imwrite(str(crops_dir / f"{label}.png"), preprocess_image(region.image, mode))
            words = cell_engine.recognize_column_words(region.image, mode=mode)
            translated = [
                type(word)(word.text, word.confidence, word.x + region.x_offset, word.y + region.y_offset, word.w, word.h)
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
        pad_x = max(2, int(cell.w * 0.06))
        pad_y = max(2, int(cell.h * 0.10))
        crop = grid.image[
            max(0, cell.y + pad_y) : min(grid.image.shape[0], cell.y + cell.h - pad_y),
            max(0, cell.x + pad_x) : min(grid.image.shape[1], cell.x + cell.w - pad_x),
        ]
        if crop.size == 0:
            continue
        raw, confidence = cell_engine.recognize_cell(crop, "integer")
        if raw.strip():
            header_cells[(lap_row, col)] = _cell_result(lap_row, col, raw, confidence)


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
