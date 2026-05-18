from __future__ import annotations

import json
from pathlib import Path

import cv2

from .cell_ocr import CellOcrResult, normalize_lap_text, parse_lap_time
from .lap_row_detector import OcrWord, VirtualLapRow, find_virtual_row
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
    image_dir = _debug_image_dir(debug_dir)
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
    cv2.imwrite(str(image_dir / "grid_parsed_overlay.png"), image)


def write_paddleocr_overlay(grid: TableGrid, words: list[OcrWord], debug_dir: Path | None) -> None:
    if debug_dir is None:
        return
    image_dir = _debug_image_dir(debug_dir)
    image = grid.image.copy()
    for word in words:
        x1, y1 = int(round(word.x)), int(round(word.y))
        x2, y2 = int(round(word.x + word.w)), int(round(word.y + word.h))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 255), 1)
        label = word.text[:16]
        cv2.putText(image, label, (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 255), 1)
    cv2.imwrite(str(image_dir / "paddleocr_overlay.png"), image)


def write_ocr_words(
    column_words: dict[int, list[OcrWord]],
    rows: list[VirtualLapRow],
    debug_dir: Path | None,
) -> None:
    if debug_dir is None:
        return
    payload = []
    for col, words in sorted(column_words.items()):
        for word in words:
            payload.append(
                {
                    "col": col,
                    "virtual_row": find_virtual_row(rows, word.center_y),
                    "text": word.text,
                    "normalized": normalize_lap_text(word.text),
                    "lap": parse_lap_time(word.text),
                    "confidence": word.confidence,
                    "x": round(word.x, 2),
                    "y": round(word.y, 2),
                    "w": round(word.w, 2),
                    "h": round(word.h, 2),
                }
            )
    with (debug_dir / "ocr_words.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_virtual_rows_overlay(
    grid: TableGrid,
    column_words: dict[int, list[OcrWord]],
    rows: list[VirtualLapRow],
    debug_dir: Path | None,
) -> None:
    if debug_dir is None:
        return
    image_dir = _debug_image_dir(debug_dir)
    image = grid.image.copy()
    for x in grid.x_lines:
        cv2.line(image, (x, 0), (x, image.shape[0] - 1), (0, 255, 0), 1)
    for row in rows:
        y = int(round(row.center_y))
        cv2.line(image, (0, y), (image.shape[1] - 1, y), (255, 0, 255), 1)
        cv2.putText(image, str(row.index), (4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    for col, words in column_words.items():
        for word in words:
            row_index = find_virtual_row(rows, word.center_y)
            x1, y1 = int(round(word.x)), int(round(word.y))
            x2, y2 = int(round(word.x + word.w)), int(round(word.y + word.h))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 255), 1)
            cv2.putText(image, f"c{col}/r{row_index or '?'}", (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 128, 255), 1)
    cv2.imwrite(str(image_dir / "virtual_rows_overlay.png"), image)


def _debug_image_dir(debug_dir: Path) -> Path:
    image_dir = debug_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def _draw_row(image, grid: TableGrid, row: int, color: tuple[int, int, int], label: str) -> None:
    if row < 0 or row >= grid.row_count:
        return
    y1 = grid.y_lines[row]
    y2 = grid.y_lines[row + 1]
    cv2.rectangle(image, (0, y1), (image.shape[1] - 1, y2), color, 2)
    cv2.putText(image, label, (8, max(y1 + 18, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
