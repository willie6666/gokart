import numpy as np

from gokart_bot.cell_ocr import normalize_kart_no, normalize_lap_text, parse_lap_time, _remove_cell_edges


def test_normalize_lap_text() -> None:
    assert parse_lap_time("1943") == 19.43
    assert parse_lap_time("19:43") == 19.43
    assert parse_lap_time("19,43") == 19.43
    assert parse_lap_time("l9.43") == 19.43
    assert parse_lap_time("O9.43") is None
    assert parse_lap_time("10.67") is None
    assert normalize_lap_text("20318") == "20.318"


def test_normalize_kart_no() -> None:
    assert normalize_kart_no("12") == 12
    assert normalize_kart_no("O7") == 7
    assert normalize_kart_no("5 77") == 5
    assert normalize_kart_no("0") is None
    assert normalize_kart_no("1000") is None


def test_remove_cell_edges_clears_left_vertical_line() -> None:
    image = np.full((40, 60), 255, dtype=np.uint8)
    image[:, :3] = 0
    image[14:30, 28:35] = 0

    result = _remove_cell_edges(image)

    assert result[:, :4].mean() > 240
    assert result[14:30, 28:35].mean() < 240
