from gokart_bot.parser import _find_date, _find_heat, _find_time


def test_find_date_standard() -> None:
    assert _find_date("Date: 2026/5/16") == "2026/5/16"
    assert _find_date("2026/5/16 Heat 12") == "2026/5/16"


def test_find_date_compact() -> None:
    assert _find_date("Date: 2026/518") == "2026/5/8"
    assert _find_date("Date: 2026/5115") == "2026/5/15"


def test_find_date_time_split_compact() -> None:
    text = "Date: 2026/ 5115 Time INE 04:02 33"
    assert _find_date(text) == "2026/5/15"
    assert _find_time(text) == "04:02:33"


def test_find_heat() -> None:
    assert _find_heat("Heat 12") == "Heat 12"
    assert _find_heat("Heat: 12") == "Heat 12"
    assert _find_heat("Heat Heat 5") == "Heat 5"


def test_find_time() -> None:
    assert _find_time("Time: 16:18:42") == "16:18:42"
    assert _find_time("Time 04:02 33") == "04:02:33"
    assert _find_time("Printed 14:30") == "14:30"


def test_find_time_hh_mm_only() -> None:
    assert _find_time("Time 12:34") == "12:34"


def test_find_date_none() -> None:
    assert _find_date("No date here") is None


def test_find_heat_none() -> None:
    assert _find_heat("No heat here") is None


def test_find_time_none() -> None:
    assert _find_time("No time here") is None
