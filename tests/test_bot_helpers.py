import pytest

from gokart_bot.bot import _parse_laps_argument


def test_parse_laps_argument_accepts_ascii_and_fullwidth_commas() -> None:
    assert _parse_laps_argument("20.1, 19.65，20.0") == [20.1, 19.65, 20.0]


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0"])
def test_parse_laps_argument_rejects_invalid_numbers(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_laps_argument(value)
