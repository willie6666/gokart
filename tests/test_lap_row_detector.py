from gokart_bot.lap_row_detector import (
    OcrWord,
    detect_virtual_lap_rows,
    fill_missing_virtual_rows,
    find_virtual_row,
    infer_lap_count,
    looks_like_lap_candidate,
)


def word(text: str, y: float) -> OcrWord:
    return OcrWord(text=text, confidence=90.0, x=10.0, y=y, w=30.0, h=10.0)


def test_detect_virtual_lap_rows_clusters_ocr_y_positions() -> None:
    rows = detect_virtual_lap_rows(
        [word("1", 100), word("19.43", 102), word("2", 130), word("19,50", 132)],
        lap_nr_y=80,
        table_height=200,
        min_y_gap=8,
    )

    assert len(rows) == 2
    assert find_virtual_row(rows, 105) == 1
    assert find_virtual_row(rows, 135) == 2


def test_looks_like_lap_candidate_variants() -> None:
    assert looks_like_lap_candidate("19.43")
    assert looks_like_lap_candidate("19,43")
    assert looks_like_lap_candidate("19:43")
    assert looks_like_lap_candidate("1943")
    assert looks_like_lap_candidate("l9.43")


def test_fill_missing_virtual_rows_uses_cluster_pitch() -> None:
    clustered = detect_virtual_lap_rows(
        [word("19.43", 100), word("19.50", 130), word("19.55", 160), word("19.60", 220)],
        lap_nr_y=80,
        table_height=260,
        min_y_gap=8,
    )
    rows = fill_missing_virtual_rows(clustered, expected_count=5, lap_nr_y=80, bottom_y=260, table_height=260)

    assert len(rows) == 5
    assert [row.source for row in rows] == ["ocr_cluster", "ocr_cluster", "ocr_cluster", "filled", "ocr_cluster"]
    assert 190 <= rows[3].center_y <= 195


def test_infer_lap_count_from_merged_lap_numbers() -> None:
    words = [word("21 18 20 19 22 23 24 25 26 27 28", 300)]

    assert infer_lap_count(words, lap_nr_y=100) == 28
