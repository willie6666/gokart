from gokart_bot.direct_ocr_parser import parse_direct_ocr_words
from gokart_bot.lap_row_detector import OcrWord


def word(text: str, x: float, y: float, w: float = 28.0, h: float = 10.0) -> OcrWord:
    return OcrWord(text=text, confidence=0.95, x=x, y=y, w=w, h=h)


def test_parse_direct_ocr_words_chains_laps_below_karts() -> None:
    parsed = parse_direct_ocr_words(
        [
            word("Date:", 0, 0),
            word("2026/5/16", 35, 0, 70),
            word("Time", 110, 0),
            word("16:18:42", 145, 0, 60),
            word("Heat", 210, 0),
            word("12", 245, 0),
            word("Lap/Nr", 10, 50, 45),
            word("7", 80, 50, 16),
            word("12", 145, 50, 18),
            word("19.50", 78, 80, 38),
            word("19.43", 80, 110, 38),
            word("20.10", 143, 80, 38),
            word("19.90", 145, 110, 38),
            word("Avg.", 10, 145, 35),
            word("19.47", 78, 145, 38),
            word("20.00", 145, 145, 38),
        ]
    )

    assert parsed.date == "2026/5/16"
    assert parsed.printed_time == "16:18:42"
    assert parsed.heat == "Heat 12"
    assert [kart.kart_no for kart in parsed.karts] == [7, 12]
    assert parsed.karts[0].laps == [19.5, 19.43]
    assert parsed.karts[0].best_lap == 19.43
    assert parsed.karts[0].avg_lap == 19.47
    assert parsed.karts[1].laps == [20.1, 19.9]
    assert parsed.karts[1].avg_lap == 20.0
    assert parsed.raw_debug_summary["mode"] == "direct_paddleocr"
    assert parsed.raw_debug_summary["kart_words"][0]["text"] == "7"
    assert [word["text"] for word in parsed.raw_debug_summary["lap_chains"][0]] == ["19.50", "19.43"]
    assert parsed.raw_debug_summary["avg_matches"][0]["text"] == "19.47"


def test_parse_direct_ocr_words_warns_without_lap_nr() -> None:
    parsed = parse_direct_ocr_words([word("19.43", 80, 110)])

    assert parsed.karts == []
    assert "Lap/Nr cell not detected" in parsed.warnings
    assert parsed.raw_debug_summary["mode"] == "direct_paddleocr"


def test_parse_direct_ocr_words_uses_nearest_next_lap_row() -> None:
    parsed = parse_direct_ocr_words(
        [
            word("Lap/Nr", 10, 50, 45),
            word("1", 100, 50, 10),
            word("21.16", 94, 80, 49),
            word("21.52", 91, 104, 48),
            word("21.24", 90, 500, 60),
            word("Avg.", 10, 540, 35),
            word("21.30", 90, 540, 48),
        ]
    )

    assert parsed.karts[0].laps[:2] == [21.16, 21.52]
    assert [word["text"] for word in parsed.raw_debug_summary["lap_chains"][0]][:2] == ["21.16", "21.52"]


def test_parse_direct_ocr_words_matches_avg_by_right_order() -> None:
    parsed = parse_direct_ocr_words(
        [
            word("Lap/Nr", 10, 50, 45),
            word("1", 100, 50, 10),
            word("7", 170, 50, 16),
            word("21.16", 94, 80, 49),
            word("21.52", 91, 104, 48),
            word("21.20", 168, 80, 49),
            word("22.52", 166, 104, 49),
            word("Avg.", 10, 540, 35),
            word("21.13", 80, 540, 48),
            word("24.93", 160, 540, 48),
        ]
    )

    assert [kart.kart_no for kart in parsed.karts] == [1, 7]
    assert [kart.avg_lap for kart in parsed.karts] == [21.13, 24.93]
    assert [word["text"] for word in parsed.raw_debug_summary["avg_matches"]] == ["21.13", "24.93"]


def test_parse_direct_ocr_words_does_not_include_avg_as_lap() -> None:
    parsed = parse_direct_ocr_words(
        [
            word("Lap/Nr", 10, 50, 45),
            word("1", 100, 50, 10),
            word("21.16", 94, 80, 49),
            word("21.52", 91, 104, 48),
            word("Avg.", 10, 130, 35),
            word("21.34", 90, 130, 48),
        ]
    )

    assert parsed.karts[0].laps == [21.16, 21.52]
    assert parsed.karts[0].avg_lap == 21.34


def test_parse_direct_ocr_words_uses_topmost_avg_value_as_lap_boundary() -> None:
    parsed = parse_direct_ocr_words(
        [
            word("Lap/Nr", 10, 50, 45),
            word("1", 100, 50, 10),
            word("21.16", 94, 80, 49),
            word("21.52", 91, 104, 48),
            word("Avg.", 10, 150, 35, 20),
            word("21.34", 90, 130, 48, 25),
        ]
    )

    assert parsed.karts[0].laps == [21.16, 21.52]
    assert parsed.karts[0].avg_lap == 21.34
