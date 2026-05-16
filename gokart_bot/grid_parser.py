from __future__ import annotations

import re

from .cell_ocr import CellOcrResult, normalize_kart_no, parse_lap_time
from .models import KartResult
from .parser import ParsedSheet, _find_date, _find_heat, _find_time
from .table_grid import TableGrid


def parse_grid_sheet(
    grid: TableGrid,
    ocr_cells: dict[tuple[int, int], CellOcrResult],
    legacy_items=None,
) -> ParsedSheet:
    sheet = ParsedSheet()
    sheet.warnings.extend(grid.warnings)
    header_text = " ".join(cell.raw_text for (row, _), cell in sorted(ocr_cells.items()) if row <= max(2, grid.row_count // 4))
    sheet.date = _find_date(header_text)
    sheet.printed_time = _find_time(header_text)
    sheet.heat = _find_heat(header_text)
    if sheet.date is None:
        sheet.warnings.append("Date not detected")
    if sheet.heat is None:
        sheet.warnings.append("Heat not detected")

    lap_position = _find_lap_nr_cell(ocr_cells)
    if lap_position is None:
        sheet.warnings.append("Lap/Nr cell not detected")
        return sheet
    kart_row, lap_col = lap_position
    avg_row = _find_avg_row(ocr_cells, kart_row + 1)
    stop_row = avg_row if avg_row is not None else grid.row_count

    results: list[KartResult] = []
    for col in range(lap_col + 1, grid.col_count):
        kart_no = normalize_kart_no(_cell_text(ocr_cells, kart_row, col))
        laps: list[float] = []
        for row in range(kart_row + 1, stop_row):
            lap = parse_lap_time(_cell_text(ocr_cells, row, col))
            if lap is not None:
                laps.append(lap)
        if kart_no is None and not laps:
            continue
        results.append(
            KartResult(
                kart_no=kart_no,
                position=col - lap_col,
                best_lap=min(laps) if laps else None,
                avg_lap=round(sum(laps) / len(laps), 3) if laps else None,
                laps=laps,
            )
        )
    sheet.karts = results
    if not results:
        sheet.warnings.append("No kart lap columns detected")
    sheet.raw_debug_summary = {
        "mode": "grid",
        "row_count": grid.row_count,
        "col_count": grid.col_count,
        "lap_nr_cell": {"row": kart_row, "col": lap_col},
        "avg_row": avg_row,
    }
    return sheet


def looks_like_lap_nr(text: str) -> bool:
    normalized = re.sub(r"[^a-z]", "", text.lower())
    return normalized in {"lapnr", "lpnr", "lapn", "lpn", "lapno", "lapnumber", "lap"}


def looks_like_avg(text: str) -> bool:
    return text.strip().lower().rstrip(".") == "avg"


def _find_lap_nr_cell(ocr_cells: dict[tuple[int, int], CellOcrResult]) -> tuple[int, int] | None:
    matches = [(row, col) for (row, col), cell in ocr_cells.items() if looks_like_lap_nr(cell.raw_text or cell.normalized_text)]
    if not matches:
        return None
    return sorted(matches, key=lambda item: (item[0], item[1]))[0]


def _find_avg_row(ocr_cells: dict[tuple[int, int], CellOcrResult], start_row: int) -> int | None:
    rows = [row for (row, _), cell in ocr_cells.items() if row >= start_row and looks_like_avg(cell.raw_text or cell.normalized_text)]
    return min(rows) if rows else None


def _cell_text(ocr_cells: dict[tuple[int, int], CellOcrResult], row: int, col: int) -> str:
    cell = ocr_cells.get((row, col))
    if cell is None:
        return ""
    return cell.normalized_text or cell.raw_text
