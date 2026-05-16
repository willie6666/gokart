from gokart_bot.cell_ocr import normalize_kart_no, normalize_lap_text, parse_lap_time
from gokart_bot.cell_ocr import CellOcrEngine
from gokart_bot.table_grid import TableGrid, build_cells
import numpy as np


def test_normalize_lap_text() -> None:
    assert parse_lap_time("1943") == 19.43
    assert parse_lap_time("19:43") == 19.43
    assert parse_lap_time("19,43") == 19.43
    assert parse_lap_time("l9.43") == 19.43
    assert parse_lap_time("O9.43") is None
    assert normalize_lap_text("20318") == "20.318"


def test_normalize_kart_no() -> None:
    assert normalize_kart_no("12") == 12
    assert normalize_kart_no("O7") == 7
    assert normalize_kart_no("5 77") == 5
    assert normalize_kart_no("0") is None
    assert normalize_kart_no("1000") is None


def test_paddle_table_ocr_maps_items_into_cells(tmp_path) -> None:
    class FakePaddle:
        def predict(self, image):
            return [
                {
                    "res": {
                        "rec_texts": ["Lap/Nr", "12", "19.43"],
                        "rec_scores": [0.95, 0.96, 0.97],
                        "rec_boxes": [[5, 5, 45, 25], [55, 5, 95, 25], [55, 35, 95, 55]],
                    }
                }
            ]

    x_lines = [0, 50, 100]
    y_lines = [0, 30, 60]
    grid = TableGrid(np.zeros((60, 100, 3), dtype=np.uint8), build_cells(x_lines, y_lines), 2, 2, x_lines, y_lines)
    engine = CellOcrEngine(paddle_ocr=FakePaddle())
    engine.has_tesseract = False

    cells = engine.recognize_grid(grid, tmp_path)

    assert cells[(0, 0)].raw_text == "Lap/Nr"
    assert cells[(0, 1)].raw_text == "12"
    assert cells[(1, 1)].raw_text == "19.43"
    assert (tmp_path / "paddle_ocr_items.json").exists()
