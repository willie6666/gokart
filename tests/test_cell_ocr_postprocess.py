from gokart_bot.cell_ocr import normalize_kart_no, normalize_lap_text, parse_lap_time


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

