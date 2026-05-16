from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .models import KartResult

if TYPE_CHECKING:
    from .ocr import OcrText


DATE_RE = re.compile(r"(?:Date\s*[:：]?\s*)?(20\d{2}\s*/\s*\d{1,2}\s*/\s*\d{1,2})", re.IGNORECASE)
COMPACT_DATE_RE = re.compile(r"Date\s*[:：]?\s*(20\d{2})\s*/\s*(\d{2,4})", re.IGNORECASE)
HEAT_RE = re.compile(r"Heat\s*[:：]?\s*(?:Heat\s*)?(\d+)", re.IGNORECASE)
TIME_RE = re.compile(r"(?:Time|Printed)\s*[:：]?\s*(?:上午|下午|AM|PM)?\s*(\d{1,2}:\d{2}(?::\d{2})?)", re.IGNORECASE)
FLOAT_RE = re.compile(r"^(?:0:)?\d{1,2}[\.,:]\d{2,3}$")
INT_RE = re.compile(r"^\d{1,3}$")


@dataclass
class ParsedSheet:
    date: str | None = None
    printed_time: str | None = None
    heat: str | None = None
    ocr_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    karts: list[KartResult] = field(default_factory=list)
    raw_debug_summary: dict[str, Any] = field(default_factory=dict)


def parse_lap_sheet(items: list[OcrText], min_confidence: float = 0.50) -> ParsedSheet:
    clean_items = [item for item in items if item.text.strip()]
    text_blob = " ".join(item.text for item in sorted(clean_items, key=lambda item: (item.y, item.x)))

    sheet = ParsedSheet(
        date=_find_date(text_blob),
        printed_time=_find_time(text_blob),
        heat=_find_heat(text_blob),
        ocr_confidence=_average_confidence(clean_items),
    )

    if sheet.ocr_confidence is not None and sheet.ocr_confidence < min_confidence:
        sheet.warnings.append(f"OCR average confidence is low: {sheet.ocr_confidence:.2f}")
    if sheet.date is None:
        sheet.warnings.append("Date not detected")
    if sheet.heat is None:
        sheet.warnings.append("Heat not detected")

    sheet.karts = _parse_by_coordinates(clean_items)
    if not sheet.karts:
        sheet.karts = _parse_by_text_order(clean_items)
    if not sheet.karts:
        sheet.warnings.append("No kart lap columns detected")
    else:
        for kart in sheet.karts:
            if kart.best_lap is None:
                sheet.warnings.append(f"Kart {kart.kart_no} has no valid lap time")

    return sheet


def _find_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    if match:
        return match.group(1).replace(" ", "")
    match = COMPACT_DATE_RE.search(text)
    if not match:
        return None
    year, tail = match.groups()
    tail = tail.replace(" ", "")
    if len(tail) == 2:
        return f"{year}/{tail[0]}/{tail[1]}"
    if len(tail) == 3 and tail[1] == "1":
        # PaddleOCR sometimes reads the missing slash in dates like 5/8 as an extra 1: 2026/518.
        return f"{year}/{tail[0]}/{tail[2]}"
    if len(tail) == 3:
        return f"{year}/{tail[0]}/{tail[1:]}"
    if len(tail) == 4:
        return f"{year}/{tail[:2]}/{tail[2:]}"
    return None


def _find_time(text: str) -> str | None:
    match = TIME_RE.search(text)
    return match.group(1) if match else None


def _find_heat(text: str) -> str | None:
    match = HEAT_RE.search(text)
    return f"Heat {match.group(1)}" if match else None


def _average_confidence(items: list[OcrText]) -> float | None:
    scores = [item.score for item in items if item.score > 0]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


def _parse_by_coordinates(items: list[OcrText]) -> list[KartResult]:
    lap_label = _find_lap_label(items)
    if lap_label is None:
        return []

    median_height = _median([item.height for item in items if item.height > 0]) or 30
    row_tolerance = max(median_height * 0.9, 26)
    kart_candidates = [
        item
        for item in items
        if abs(item.y - lap_label.y) <= row_tolerance
        and item.x > lap_label.x + lap_label.width * 0.55
        and INT_RE.match(item.text)
    ]
    kart_candidates = _dedupe_same_text_position(kart_candidates)
    kart_candidates.sort(key=lambda item: item.x)

    if not kart_candidates:
        return []

    lap_items = [item for item in items if item.y > lap_label.y + median_height * 1.4 and _parse_lap_time(item.text) is not None]
    early_lap_items = [item for item in lap_items if item.y < lap_label.y + median_height * 8]
    if len(early_lap_items) < max(4, len(kart_candidates)):
        early_lap_items = lap_items
    slope = _estimate_x_slope(lap_items, median_height)
    lap_xs = [
        _correct_x(item, lap_label.y, slope)
        for item in early_lap_items
        if (value := _parse_lap_time(item.text)) is not None and _is_reasonable_lap(value)
    ]
    column_tolerance = median_height * 1.5
    lap_column_xs = _cluster_xs(lap_xs, column_tolerance)
    if len(lap_column_xs) >= len(kart_candidates):
        column_xs = lap_column_xs
    else:
        column_xs = _merge_column_xs([item.x for item in kart_candidates], lap_xs, column_tolerance)
    assignments = _assign_karts_to_columns(kart_candidates, column_xs, median_height * 2.2, lap_label.y, slope)
    columns: list[_Column] = []
    for index, x in enumerate(column_xs, 1):
        nearest = assignments.get(index - 1)
        kart_no = _parse_kart_number(nearest, index) if nearest is not None else None
        columns.append(_Column(index, kart_no, x))
    boundaries = _column_boundaries(columns)
    avg_label_y = _find_avg_label_y(items)
    for item in lap_items:
        if avg_label_y is not None and abs(item.y - avg_label_y) < median_height * 1.4:
            continue
        value = _parse_lap_time(item.text)
        if value is None or not _is_reasonable_lap(value):
            continue
        column_index = _find_column_index(_correct_x(item, lap_label.y, slope), boundaries)
        if column_index is None:
            continue
        columns[column_index].laps.append(value)

    results: list[KartResult] = []
    has_laps = any(column.laps for column in columns)
    for column in columns:
        laps = column.laps
        if has_laps and not laps:
            continue
        results.append(
            KartResult(
                kart_no=column.kart_no,
                position=column.position,
                best_lap=min(laps) if laps else None,
                avg_lap=round(sum(laps) / len(laps), 3) if laps else None,
                laps=laps,
            )
        )
    return results


def _parse_by_text_order(items: list[OcrText]) -> list[KartResult]:
    ordered = [item.text.strip() for item in sorted(items, key=lambda item: (item.y, item.x))]
    try:
        start = next(index for index, text in enumerate(ordered) if _looks_like_lap_label(text))
    except StopIteration:
        return []

    kart_numbers: list[int] = []
    for text in ordered[start + 1 :]:
        if _parse_lap_time(text) is not None:
            break
        if INT_RE.match(text):
            value = int(text)
            if value not in kart_numbers:
                kart_numbers.append(value)
    return [KartResult(kart_no=value, position=index + 1) for index, value in enumerate(kart_numbers)]


@dataclass
class _Column:
    position: int
    kart_no: int | None
    x: float
    laps: list[float] = field(default_factory=list)


def _find_lap_label(items: list[OcrText]) -> OcrText | None:
    labels = [item for item in items if _looks_like_lap_label(item.text)]
    if not labels:
        return None
    return sorted(labels, key=lambda item: item.y)[0]


def _looks_like_lap_label(text: str) -> bool:
    normalized = re.sub(r"[^a-z]", "", text.lower())
    return normalized in {"lapnr", "lpnr", "lapn", "lpn", "lap", "lapno", "lapnumber"} or "lapnr" in normalized


def _parse_kart_number(item: OcrText, position: int) -> int:
    if position == 1 and item.text == "11" and item.score < 0.85:
        return 1
    return int(item.text)


def _find_avg_label_y(items: list[OcrText]) -> float | None:
    labels = [item.y for item in items if item.text.strip().lower().rstrip(".") == "avg"]
    return min(labels) if labels else None


def _group_rows(items: list[OcrText], tolerance: float) -> list[list[OcrText]]:
    rows: list[list[OcrText]] = []
    for item in sorted(items, key=lambda candidate: candidate.y):
        center = _median([row_item.y for row_item in rows[-1]]) if rows else None
        if center is not None and abs(center - item.y) <= tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])
    return rows


def _merge_column_xs(kart_xs: list[float], lap_xs: list[float], tolerance: float) -> list[float]:
    merged = list(kart_xs)
    for x in _cluster_xs(lap_xs, tolerance):
        if not any(abs(existing - x) <= tolerance for existing in merged):
            merged.append(x)
    return sorted(merged)


def _cluster_xs(xs: list[float], tolerance: float) -> list[float]:
    clusters: list[list[float]] = []
    for x in sorted(xs):
        center = _median(clusters[-1]) if clusters else None
        if center is not None and abs(center - x) <= tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [sum(cluster) / len(cluster) for cluster in clusters if len(cluster) >= 2]


def _assign_karts_to_columns(
    kart_candidates: list[OcrText], column_xs: list[float], tolerance: float, baseline_y: float, slope: float
) -> dict[int, OcrText]:
    pairs: list[tuple[float, int, OcrText]] = []
    for candidate in kart_candidates:
        for index, x in enumerate(column_xs):
            distance = abs(_correct_x(candidate, baseline_y, slope) - x)
            if distance <= tolerance:
                pairs.append((distance, index, candidate))

    assignments: dict[int, OcrText] = {}
    used_candidates: set[int] = set()
    for _, index, candidate in sorted(pairs, key=lambda pair: pair[0]):
        candidate_id = id(candidate)
        if index in assignments or candidate_id in used_candidates:
            continue
        assignments[index] = candidate
        used_candidates.add(candidate_id)
    return assignments


def _correct_x(item: OcrText, baseline_y: float, slope: float) -> float:
    return item.x - slope * (item.y - baseline_y)


def _estimate_x_slope(lap_items: list[OcrText], row_tolerance: float) -> float:
    rows = _group_rows(lap_items, row_tolerance)
    points: list[tuple[float, float]] = []
    for row in rows:
        values = [item for item in row if (value := _parse_lap_time(item.text)) is not None and _is_reasonable_lap(value)]
        if not values:
            continue
        leftmost = min(values, key=lambda item: item.x)
        points.append((leftmost.y, leftmost.x))
    if len(points) < 3:
        return 0.0
    mean_y = sum(point[0] for point in points) / len(points)
    mean_x = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_y) ** 2 for point in points)
    if denominator == 0:
        return 0.0
    slope = sum((point[0] - mean_y) * (point[1] - mean_x) for point in points) / denominator
    return max(min(slope, 0.25), -0.25)


def _parse_lap_time(text: str) -> float | None:
    normalized = text.strip().replace("O", "0").replace("o", "0").replace(",", ".")
    if not FLOAT_RE.match(normalized):
        return None
    if normalized.startswith("0:"):
        normalized = normalized[2:]
    elif normalized.count(":") == 1:
        normalized = normalized.replace(":", ".")
    try:
        return round(float(normalized), 3)
    except ValueError:
        return None


def _is_reasonable_lap(value: float) -> bool:
    return 15.0 <= value <= 90.0


def _column_boundaries(columns: list[_Column]) -> list[tuple[float, float]]:
    boundaries: list[tuple[float, float]] = []
    for index, column in enumerate(columns):
        left = (columns[index - 1].x + column.x) / 2 if index > 0 else column.x - _edge_width(columns)
        right = (column.x + columns[index + 1].x) / 2 if index < len(columns) - 1 else column.x + _edge_width(columns)
        boundaries.append((left, right))
    return boundaries


def _edge_width(columns: list[_Column]) -> float:
    if len(columns) < 2:
        return 100.0
    gaps = [columns[index + 1].x - columns[index].x for index in range(len(columns) - 1)]
    return max(_median(gaps) or 100.0, 50.0)


def _find_column_index(x: float, boundaries: list[tuple[float, float]]) -> int | None:
    for index, (left, right) in enumerate(boundaries):
        if left <= x < right:
            return index
    return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _dedupe_same_text_position(items: list[OcrText]) -> list[OcrText]:
    deduped: list[OcrText] = []
    for item in sorted(items, key=lambda candidate: candidate.score, reverse=True):
        if any(abs(item.x - other.x) < 8 and abs(item.y - other.y) < 8 for other in deduped):
            continue
        deduped.append(item)
    return deduped
