from pathlib import Path
import json

import pytest

from gokart_bot.table_grid import extract_table_grid


@pytest.mark.parametrize(
    "expected_name",
    ["heat12.json", "heat15.json"],
)
def test_sample_image_grid_artifacts(expected_name: str, tmp_path: Path) -> None:
    expected_path = Path("tests/fixtures/expected") / expected_name
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    image_path = Path(expected["image"])
    if not image_path.exists():
        pytest.skip(f"sample image not available: {image_path}")
    grid = extract_table_grid(image_path, tmp_path / expected_name, max_side=1600)

    if expected.get("requires_warning"):
        assert grid.warnings
    else:
        assert grid.col_count >= expected["min_columns"]
        assert grid.row_count >= expected["min_rows"]
    assert (tmp_path / expected_name / "images" / "table_warped.png").exists()
    assert (tmp_path / expected_name / "images" / "grid_overlay.png").exists()
