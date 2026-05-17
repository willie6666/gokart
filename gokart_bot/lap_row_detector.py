from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import mean
from typing import Literal


@dataclass(frozen=True)
class TesseractWord:
    text: str
    confidence: float | None
    x: float
    y: float
    w: float
    h: float

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2


@dataclass(frozen=True)
class VirtualLapRow:
    index: int
    center_y: float
    top: float
    bottom: float
    source: Literal["ocr_cluster", "estimated"]


def detect_virtual_lap_rows(
    words: list[TesseractWord],
    lap_nr_y: float,
    table_height: int,
    min_y_gap: float = 8.0,
    bottom_y: float | None = None,
    expected_count: int | None = None,
) -> list[VirtualLapRow]:
    if expected_count and expected_count >= 2:
        return _estimated_rows(lap_nr_y, bottom_y or float(table_height), expected_count)

    candidates = [word.center_y for word in words if word.center_y > lap_nr_y and _looks_like_row_anchor(word.text)]
    clusters = cluster_y_positions(candidates, min_y_gap)
    centers = [mean(cluster) for cluster in clusters]
    if not centers:
        return []

    rows: list[VirtualLapRow] = []
    for index, center in enumerate(centers):
        if index == 0:
            gap = centers[1] - center if len(centers) > 1 else 24.0
            top = max(lap_nr_y, center - gap / 2)
        else:
            top = (centers[index - 1] + center) / 2
        if index + 1 < len(centers):
            bottom = (center + centers[index + 1]) / 2
        else:
            gap = center - centers[index - 1] if index > 0 else 24.0
            bottom = min(float(table_height), center + gap / 2)
        rows.append(VirtualLapRow(index=index + 1, center_y=center, top=top, bottom=bottom, source="ocr_cluster"))
    return rows


def infer_lap_count(words: list[TesseractWord], lap_nr_y: float, max_laps: int = 80) -> int | None:
    values: list[int] = []
    for word in words:
        if word.center_y <= lap_nr_y:
            continue
        for token in re.findall(r"\d{1,2}", word.text):
            value = int(token)
            if 1 <= value <= max_laps:
                values.append(value)
    if not values:
        return None
    best = max(values)
    return best if best >= 2 else None


def cluster_y_positions(values: list[float], tolerance: float) -> list[list[float]]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if clusters and abs(value - mean(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def find_virtual_row(rows: list[VirtualLapRow], y: float) -> int | None:
    for row in rows:
        if row.top <= y < row.bottom:
            return row.index
    return None


def _estimated_rows(top_y: float, bottom_y: float, count: int) -> list[VirtualLapRow]:
    if bottom_y <= top_y:
        return []
    step = (bottom_y - top_y) / count
    rows = []
    for index in range(count):
        top = top_y + index * step
        bottom = top_y + (index + 1) * step
        rows.append(VirtualLapRow(index=index + 1, center_y=(top + bottom) / 2, top=top, bottom=bottom, source="estimated"))
    return rows


def looks_like_lap_candidate(text: str) -> bool:
    normalized = _normalize_lap_text(text)
    if re.fullmatch(r"\d{4,5}", normalized):
        return True
    if not re.fullmatch(r"\d{1,2}\.\d{2,3}", normalized):
        return False
    value = float(normalized)
    return 10.0 <= value <= 90.0


def _looks_like_row_anchor(text: str) -> bool:
    stripped = text.strip()
    if looks_like_lap_candidate(stripped):
        return True
    return re.fullmatch(r"\d{1,2}", stripped) is not None


def _normalize_lap_text(text: str) -> str:
    text = text.strip()
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1").replace("|", "1")
    text = text.replace(",", ".").replace(":", ".")
    text = re.sub(r"[^0-9.]", "", text)
    if re.fullmatch(r"\d{4}", text):
        return f"{text[:2]}.{text[2:]}"
    if re.fullmatch(r"\d{5}", text):
        return f"{text[:2]}.{text[2:]}"
    return text
