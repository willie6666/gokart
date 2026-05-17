import numpy as np

from gokart_bot.cell_ocr import CellOcrResult
from gokart_bot.grid_parser import parse_grid_sheet
from gokart_bot.table_grid import TableGrid, build_cells


def cell(row: int, col: int, text: str) -> CellOcrResult:
    return CellOcrResult(row, col, text, text, 0.9)


def grid() -> TableGrid:
    x_lines = [index * 10 for index in range(8)]
    y_lines = [index * 10 for index in range(8)]
    return TableGrid(np.zeros((70, 70, 3), dtype=np.uint8), build_cells(x_lines, y_lines), 7, 7, x_lines, y_lines)


def test_parse_grid_sheet_by_row_col() -> None:
    cells = {
        (0, 0): cell(0, 0, "Date: 2026/5/16 Heat 12 Time 16:18:42"),
        (2, 1): cell(2, 1, "Lap/Nr"),
        (2, 2): cell(2, 2, "12"),
        (2, 3): cell(2, 3, "7"),
        (3, 2): cell(3, 2, "20.84"),
        (3, 3): cell(3, 3, "21.62"),
        (4, 2): cell(4, 2, "19:65"),
        (4, 3): cell(4, 3, "20,18"),
        (5, 1): cell(5, 1, "Avg."),
    }

    parsed = parse_grid_sheet(grid(), cells)

    assert parsed.date == "2026/5/16"
    assert parsed.heat == "Heat 12"
    assert [kart.kart_no for kart in parsed.karts[:2]] == [12, 7]
    assert parsed.karts[0].laps == [20.84, 19.65]
    assert parsed.karts[1].best_lap == 20.18


def test_unknown_kart_number_is_not_auto_filled() -> None:
    cells = {
        (2, 1): cell(2, 1, "Lp/Nr"),
        (2, 3): cell(2, 3, "2"),
        (3, 2): cell(3, 2, "19.42"),
        (3, 3): cell(3, 3, "24.26"),
        (4, 2): cell(4, 2, "19.57"),
        (4, 3): cell(4, 3, "24.58"),
    }

    parsed = parse_grid_sheet(grid(), cells)

    assert [kart.kart_no for kart in parsed.karts[:2]] == [None, 2]
    assert parsed.karts[0].best_lap == 19.42


def test_parse_grid_sheet_ignores_single_lap_noise_columns() -> None:
    cells = {
        (2, 1): cell(2, 1, "Lap/Nr"),
        (2, 2): cell(2, 2, "3"),
        (2, 3): cell(2, 3, "7"),
        (3, 2): cell(3, 2, "20.02"),
        (4, 2): cell(4, 2, "19.69"),
        (3, 3): cell(3, 3, "21.11"),
    }

    parsed = parse_grid_sheet(grid(), cells)

    assert [kart.kart_no for kart in parsed.karts] == [3]


def test_parse_grid_sheet_drops_obvious_lap_outliers() -> None:
    cells = {
        (2, 1): cell(2, 1, "Lap/Nr"),
        (2, 2): cell(2, 2, "5"),
        (3, 2): cell(3, 2, "24.63"),
        (4, 2): cell(4, 2, "23.29"),
        (5, 2): cell(5, 2, "22.88"),
        (6, 2): cell(6, 2, "22.47"),
        (7, 2): cell(7, 2, "70.05"),
    }

    parsed = parse_grid_sheet(grid(), cells)

    assert parsed.karts[0].laps == [24.63, 23.29, 22.88, 22.47]


def test_parse_grid_sheet_extracts_multiple_laps_from_merged_cell() -> None:
    cells = {
        (2, 1): cell(2, 1, "Lap/Nr"),
        (2, 2): cell(2, 2, "5"),
        (3, 2): cell(3, 2, "24.63"),
        (4, 2): cell(4, 2, "22.17"),
        (5, 2): cell(5, 2, "24.01 23.67 26.03"),
    }

    parsed = parse_grid_sheet(grid(), cells)

    assert parsed.karts[0].best_lap == 22.17
    assert parsed.karts[0].laps == [24.63, 22.17, 24.01, 23.67, 26.03]


def test_first_kart_eleven_before_two_is_corrected_to_one() -> None:
    cells = {
        (2, 1): cell(2, 1, "Lap/Nr"),
        (2, 2): cell(2, 2, "11"),
        (2, 3): cell(2, 3, "2"),
        (3, 2): cell(3, 2, "19.43"),
        (4, 2): cell(4, 2, "19.65"),
        (3, 3): cell(3, 3, "25.66"),
        (4, 3): cell(4, 3, "24.56"),
    }

    parsed = parse_grid_sheet(grid(), cells)

    assert [kart.kart_no for kart in parsed.karts] == [1, 2]
