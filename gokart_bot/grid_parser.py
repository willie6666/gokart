from __future__ import annotations

import re

from .cell_ocr import CellOcrResult, parse_kart_no, parse_lap_time
from .lap_row_detector import OcrWord, detect_virtual_lap_rows, fill_missing_virtual_rows, find_virtual_row, infer_lap_count
from .models import KartResult
from .parser import ParsedSheet, _find_date, _find_heat, _find_time
from .table_grid import TableGrid


AVG_MISMATCH_TOLERANCE = 0.05


def parse_grid_sheet(
    grid: TableGrid,
    ocr_cells: dict[tuple[int, int], CellOcrResult],
) -> ParsedSheet:
    sheet = ParsedSheet()
    sheet.warnings.extend(grid.warnings)
    _parse_header(sheet, grid, ocr_cells)

    lap_position = _find_lap_nr_cell(ocr_cells)
    if lap_position is None:
        sheet.warnings.append("Lap/Nr cell not detected")
        return sheet
    kart_row, lap_col = lap_position
    avg_row = _find_avg_row(ocr_cells, kart_row + 1)
    stop_row = avg_row if avg_row is not None else grid.row_count

    results: list[KartResult] = []
    for col in range(lap_col + 1, grid.col_count):
        kart_no = _kart_no_from_header_cells(ocr_cells, kart_row, col)
        laps: list[float] = []
        for row in range(kart_row + 1, stop_row):
            laps.extend(_parse_lap_times(_cell_text(ocr_cells, row, col)))
        laps = _drop_column_outliers(laps)
        if len(laps) < 2:
            continue
        results.append(
            _build_kart_result(
                sheet,
                kart_no=kart_no,
                position=col - lap_col,
                laps=laps,
                printed_avg=_avg_lap_from_cells(ocr_cells, avg_row, col),
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


def parse_grid_sheet_with_virtual_rows(
    grid: TableGrid,
    header_cells: dict[tuple[int, int], CellOcrResult],
    column_words: dict[int, list[OcrWord]],
) -> ParsedSheet:
    sheet = ParsedSheet()
    sheet.warnings.extend(grid.warnings)
    _parse_header(sheet, grid, header_cells)

    lap_position = _find_lap_nr_cell(header_cells)
    if lap_position is None:
        sheet.warnings.append("Lap/Nr cell not detected")
        return sheet
    kart_row, lap_col = lap_position
    avg_row = _find_avg_row(header_cells, kart_row + 1)
    lap_cell = grid.cell(kart_row, lap_col)
    lap_nr_y = float(lap_cell.y + lap_cell.h) if lap_cell else float(grid.y_lines[min(kart_row + 1, len(grid.y_lines) - 1)])
    avg_y = _find_avg_y(grid, header_cells, kart_row + 1)

    all_words = [word for words in column_words.values() for word in words]
    expected_laps = infer_lap_count(column_words.get(lap_col, []), lap_nr_y) or _infer_lap_count_from_cells(header_cells, kart_row + 1, lap_col)
    clustered_rows = detect_virtual_lap_rows(all_words, lap_nr_y=lap_nr_y, table_height=int(avg_y or grid.image.shape[0]))
    if expected_laps is not None and len(clustered_rows) > expected_laps:
        rows = clustered_rows[:expected_laps]
    elif expected_laps is not None and expected_laps >= len(clustered_rows) + 3:
        rows = fill_missing_virtual_rows(clustered_rows, expected_laps, lap_nr_y, avg_y, int(avg_y or grid.image.shape[0]))
    else:
        rows = clustered_rows
    if not rows:
        sheet.warnings.append("No virtual lap rows detected")
        return sheet

    results: list[KartResult] = []
    for col in sorted(column_words):
        if col <= lap_col:
            continue
        kart_no = _kart_no_from_header_cells(header_cells, kart_row, col)
        bucket: dict[int, list[tuple[float, float | None, str]]] = {}
        for word in column_words[col]:
            if avg_y is not None and word.center_y >= avg_y:
                continue
            row_index = find_virtual_row(rows, word.center_y)
            if row_index is None:
                continue
            for lap in _parse_lap_times(word.text):
                bucket.setdefault(row_index, []).append((lap, word.confidence, word.text))
        laps = [_choose_lap(candidates) for _, candidates in sorted(bucket.items())]
        laps = [lap for lap in laps if lap is not None]
        laps = _drop_column_outliers(laps)
        if len(laps) < 2:
            continue
        results.append(
            _build_kart_result(
                sheet,
                kart_no=kart_no,
                position=col - lap_col,
                laps=laps,
                printed_avg=_avg_lap_from_cells(header_cells, avg_row, col),
            )
        )

    sheet.karts = results
    if not results:
        sheet.warnings.append("No kart lap columns detected")
    sheet.raw_debug_summary = {
        "mode": "grid_virtual_rows",
        "row_count": grid.row_count,
        "col_count": grid.col_count,
        "lap_nr_cell": {"row": kart_row, "col": lap_col},
        "virtual_rows": [row.__dict__ for row in rows],
        "expected_laps": expected_laps,
        "clustered_laps": len(clustered_rows),
        "avg_y": avg_y,
        "avg_row": avg_row,
    }
    return sheet


def looks_like_lap_nr(text: str) -> bool:
    return re.fullmatch(r"\s*Lap\s*/\s*N[ro]\.?\s*", text, re.IGNORECASE) is not None


def looks_like_avg(text: str) -> bool:
    return re.fullmatch(r"\s*Avg\.?\s*", text, re.IGNORECASE) is not None


def _parse_header(sheet: ParsedSheet, grid: TableGrid, cells: dict[tuple[int, int], CellOcrResult]) -> None:
    header_text = " ".join(cell.raw_text for (row, _), cell in sorted(cells.items()) if row <= max(2, grid.row_count // 4))
    sheet.date = _find_date(header_text)
    sheet.printed_time = _find_time(header_text)
    sheet.heat = _find_heat(header_text)
    if sheet.date is None:
        sheet.warnings.append("Date not detected")
    if sheet.heat is None:
        sheet.warnings.append("Heat not detected")


def _find_lap_nr_cell(ocr_cells: dict[tuple[int, int], CellOcrResult]) -> tuple[int, int] | None:
    matches = [(row, col) for (row, col), cell in ocr_cells.items() if looks_like_lap_nr(cell.raw_text)]
    if not matches:
        return None
    return sorted(matches, key=lambda item: (item[0], item[1]))[0]


def _find_avg_row(ocr_cells: dict[tuple[int, int], CellOcrResult], start_row: int) -> int | None:
    rows = [row for (row, _), cell in ocr_cells.items() if row >= start_row and looks_like_avg(cell.raw_text)]
    return min(rows) if rows else None


def _find_avg_y(grid: TableGrid, ocr_cells: dict[tuple[int, int], CellOcrResult], start_row: int) -> float | None:
    avg_row = _find_avg_row(ocr_cells, start_row)
    if avg_row is None:
        return None
    cell = grid.cell(avg_row, 0)
    return float(cell.y) if cell else None


def _infer_lap_count_from_cells(ocr_cells: dict[tuple[int, int], CellOcrResult], start_row: int, lap_col: int) -> int | None:
    values: list[int] = []
    for (row, col), cell in ocr_cells.items():
        if row < start_row or col != lap_col:
            continue
        for token in re.findall(r"\d{1,2}", cell.raw_text):
            value = int(token)
            if 1 <= value <= 80:
                values.append(value)
    if not values:
        return None
    best = max(values)
    return best if best >= 2 else None


def _cell_text(ocr_cells: dict[tuple[int, int], CellOcrResult], row: int, col: int) -> str:
    cell = ocr_cells.get((row, col))
    if cell is None:
        return ""
    return cell.raw_text


def _kart_no_from_header_cells(ocr_cells: dict[tuple[int, int], CellOcrResult], row: int, col: int) -> int | None:
    return parse_kart_no(_cell_text(ocr_cells, row, col))


def _avg_lap_from_cells(ocr_cells: dict[tuple[int, int], CellOcrResult], avg_row: int | None, col: int) -> float | None:
    if avg_row is None:
        return None
    values = _parse_lap_times(_cell_text(ocr_cells, avg_row, col))
    return values[0] if values else None


def _build_kart_result(
    sheet: ParsedSheet,
    kart_no: int | None,
    position: int,
    laps: list[float],
    printed_avg: float | None,
) -> KartResult:
    computed_avg = round(sum(laps) / len(laps), 3) if laps else None
    if printed_avg is not None and computed_avg is not None and abs(printed_avg - computed_avg) > AVG_MISMATCH_TOLERANCE:
        label = f"車號 {kart_no}" if kart_no is not None else f"欄位 {position}"
        sheet.warnings.append(f"{label} 平均圈速不一致：表格 {printed_avg:.2f}s，辨識圈速平均 {computed_avg:.2f}s，請檢查上方圈速")
    return KartResult(
        kart_no=kart_no,
        position=position,
        best_lap=min(laps) if laps else None,
        avg_lap=printed_avg,
        laps=laps,
    )


def _drop_column_outliers(laps: list[float]) -> list[float]:
    return laps


def _parse_lap_times(text: str) -> list[float]:
    lap = parse_lap_time(text)
    if lap is not None:
        return [lap]
    laps: list[float] = []
    pattern = r"\d{1,2}[:：]\d{1,2}\.\d{2,3}|\d{1,2}\.\d{2,3}"
    for token in re.findall(pattern, text):
        lap = parse_lap_time(token)
        if lap is not None:
            laps.append(lap)
    return laps


def _choose_lap(candidates: list[tuple[float, float | None, str]]) -> float | None:
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda item: ((item[1] or 0), -abs(len(item[2]) - 5)), reverse=True)
    return candidates[0][0]
