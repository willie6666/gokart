from gokart_bot.cell_ocr import parse_kart_no, parse_lap_time


def test_parse_lap_time() -> None:
    assert parse_lap_time("19.43") == 19.43
    assert parse_lap_time("1:19.43") == 79.43
    assert parse_lap_time("1943") is None
    assert parse_lap_time("19:43") is None
    assert parse_lap_time("19,43") is None
    assert parse_lap_time("l9.43") is None
    assert parse_lap_time("O9.43") is None
    assert parse_lap_time("10.67") is None


def test_parse_kart_no() -> None:
    assert parse_kart_no("12") == 12
    assert parse_kart_no("O7") is None
    assert parse_kart_no("5 77") is None
    assert parse_kart_no("0") is None
    assert parse_kart_no("1000") is None
