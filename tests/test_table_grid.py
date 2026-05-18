from pathlib import Path

import cv2
import numpy as np
import pytest

from gokart_bot.table_grid import build_cells, extract_table_grid, merge_positions


def test_merge_positions() -> None:
    assert merge_positions([1, 2, 3, 20, 21], 2) == [2, 20]


def test_build_cells() -> None:
    cells = build_cells([0, 10, 20], [0, 5, 10])
    assert len(cells) == 4
    assert cells[0].row == 0
    assert cells[0].col == 0
    assert cells[0].w == 10


def test_incomplete_synthetic_table_fails_without_fallback(tmp_path: Path) -> None:
    image = np.full((500, 1100, 3), 255, dtype=np.uint8)
    for x in range(20, 781, 40):
        cv2.line(image, (x, 60), (x, 460), (0, 0, 0), 2)
    for y in range(60, 461, 40):
        cv2.line(image, (20, y), (620, y), (0, 0, 0), 2)
    for x in range(700, 861, 80):
        cv2.line(image, (x, 80), (x, 260), (0, 0, 0), 2)
    for y in range(80, 261, 45):
        cv2.line(image, (700, y), (860, y), (0, 0, 0), 2)
    path = tmp_path / "synthetic.jpg"
    cv2.imwrite(str(path), image)

    with pytest.raises(RuntimeError, match="無法辨識"):
        extract_table_grid(path, tmp_path / "debug", max_side=900)


def test_partial_table_sample_fails_without_fallback(tmp_path: Path) -> None:
    image_path = Path("examples/IMG_20260508_151106.jpg")
    if not image_path.exists():
        pytest.skip(f"sample image not available: {image_path}")

    with pytest.raises(RuntimeError, match="無法辨識"):
        extract_table_grid(image_path, tmp_path / "partial", max_side=1600)
