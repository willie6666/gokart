from gokart_bot.ocr import OcrText
from gokart_bot.parser import parse_lap_sheet


def item(text: str, x: float, y: float, score: float = 0.95) -> OcrText:
    return OcrText(text=text, score=score, box=[(x - 10, y - 5), (x + 10, y - 5), (x + 10, y + 5), (x - 10, y + 5)])


def test_parse_lap_sheet_by_coordinates() -> None:
    items = [
        item("Date:", 20, 20),
        item("2026/5/16", 90, 20),
        item("Heat:", 180, 20),
        item("Heat 12", 240, 20),
        item("Lap/Nr", 20, 100),
        item("12", 100, 100),
        item("7", 180, 100),
        item("1", 20, 140),
        item("20.84", 100, 140),
        item("21.62", 180, 140),
        item("2", 20, 170),
        item("19.65", 100, 170),
        item("20.18", 180, 170),
    ]

    parsed = parse_lap_sheet(items)

    assert parsed.date == "2026/5/16"
    assert parsed.heat == "Heat 12"
    assert len(parsed.karts) == 2
    assert parsed.karts[0].kart_no == 12
    assert parsed.karts[0].best_lap == 19.65
    assert parsed.karts[1].kart_no == 7
    assert parsed.karts[1].best_lap == 20.18


def test_parse_lap_sheet_falls_back_to_kart_numbers() -> None:
    items = [
        item("Lap/Nr", 20, 100),
        item("3", 100, 100),
        item("8", 180, 100),
        item("20.70", 100, 140),
    ]

    parsed = parse_lap_sheet(items)

    assert [kart.kart_no for kart in parsed.karts] == [3]


def test_parse_compact_ocr_date_and_low_confidence_first_kart() -> None:
    items = [
        item("Date: 2026/518", 90, 20),
        item("Lap/Nr", 20, 100),
        item("11", 100, 100, score=0.69),
        item("7", 180, 100),
        item("20.22", 100, 140),
        item("20.42", 180, 140),
    ]

    parsed = parse_lap_sheet(items)

    assert parsed.date == "2026/5/8"
    assert [kart.kart_no for kart in parsed.karts] == [1, 7]


def test_missing_kart_number_becomes_unknown_column() -> None:
    items = [
        item("Lap/Nr", 20, 100),
        item("3", 100, 100),
        item("8", 200, 100),
        item("5", 400, 100),
        item("20.82", 100, 150),
        item("19.69", 200, 150),
        item("21.26", 300, 150),
        item("21.84", 400, 150),
        item("20.90", 100, 170),
        item("19.87", 200, 170),
        item("21.33", 300, 170),
        item("22.17", 400, 170),
    ]

    parsed = parse_lap_sheet(items)

    assert [kart.kart_no for kart in parsed.karts] == [3, 8, None, 5]
    assert parsed.karts[2].best_lap == 21.26


def test_does_not_infer_missing_first_kart_when_second_is_two() -> None:
    items = [
        item("Lp/Nr", 20, 100),
        item("2", 200, 100),
        item("19.42", 100, 150),
        item("24.26", 200, 150),
        item("19.57", 100, 170),
        item("24.58", 200, 170),
    ]

    parsed = parse_lap_sheet(items)

    assert [kart.kart_no for kart in parsed.karts] == [None, 2]
    assert parsed.karts[0].best_lap == 19.42
