from __future__ import annotations

import json
from pathlib import Path

import cv2

from .cell_ocr import CellOcrResult
from .parser import ParsedSheet
from .table_grid import TableGrid


def write_cell_ocr(cells: dict[tuple[int, int], CellOcrResult], debug_dir: Path | None) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = [cell.to_dict() for _, cell in sorted(cells.items())]
    with (debug_dir / "cell_ocr.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_debug_parsed(parsed: ParsedSheet, debug_dir: Path | None) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": parsed.date,
        "printed_time": parsed.printed_time,
        "heat": parsed.heat,
        "ocr_confidence": parsed.ocr_confidence,
        "warnings": parsed.warnings,
        "karts": [kart.to_dict() for kart in parsed.karts],
        "raw_debug_summary": getattr(parsed, "raw_debug_summary", None),
    }
    with (debug_dir / "parsed.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_parsed_overlay(grid: TableGrid, parsed: ParsedSheet, debug_dir: Path | None) -> None:
    if debug_dir is None:
        return
    image = grid.image.copy()
    for x in grid.x_lines:
        cv2.line(image, (x, 0), (x, image.shape[0] - 1), (0, 255, 0), 1)
    for y in grid.y_lines:
        cv2.line(image, (0, y), (image.shape[1] - 1, y), (0, 255, 0), 1)
    summary = parsed.raw_debug_summary or {}
    lap_cell = summary.get("lap_nr_cell") or {}
    lap_row = lap_cell.get("row")
    lap_col = lap_cell.get("col")
    avg_row = summary.get("avg_row")
    if isinstance(lap_row, int):
        _draw_row(image, grid, lap_row, (255, 0, 0), "kart row")
    if isinstance(avg_row, int):
        _draw_row(image, grid, avg_row, (0, 128, 255), "avg row")
    if isinstance(lap_row, int) and isinstance(lap_col, int):
        cell = grid.cell(lap_row, lap_col)
        if cell:
            cv2.rectangle(image, (cell.x, cell.y), (cell.x + cell.w, cell.y + cell.h), (0, 0, 255), 3)
            cv2.putText(image, "Lap/Nr", (cell.x + 4, cell.y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(str(debug_dir / "grid_overlay.png"), image)


def _draw_row(image, grid: TableGrid, row: int, color: tuple[int, int, int], label: str) -> None:
    if row < 0 or row >= grid.row_count:
        return
    y1 = grid.y_lines[row]
    y2 = grid.y_lines[row + 1]
    cv2.rectangle(image, (0, y1), (image.shape[1] - 1, y2), color, 2)
    cv2.putText(image, label, (8, max(y1 + 18, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
