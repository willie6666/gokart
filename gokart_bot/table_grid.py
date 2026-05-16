from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .image_preprocess import make_line_mask, make_line_masks, resize_max_side, warp_perspective


@dataclass(frozen=True)
class CellBox:
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class TableGrid:
    image: np.ndarray
    cells: list[CellBox]
    row_count: int
    col_count: int
    x_lines: list[int]
    y_lines: list[int]
    warnings: list[str] = field(default_factory=list)

    def cell(self, row: int, col: int) -> CellBox | None:
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None


def extract_table_grid(image_path: Path, debug_dir: Path | None = None, max_side: int = 2200) -> TableGrid:
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Cannot read image: {image_path}")
    original = resize_max_side(original, max_side)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "original.jpg"), original)

    scanned, scan_warnings = scan_document(original, debug_dir)
    table, warnings = find_and_warp_main_table(scanned, debug_dir)
    warnings = scan_warnings + warnings
    x_lines, y_lines = detect_grid_lines(table, debug_dir)
    col_count = len(x_lines) - 1
    row_count = len(y_lines) - 1
    if col_count < 6:
        warnings.append(f"grid has too few columns: {col_count}")
    if row_count < 10:
        warnings.append(f"grid has too few rows: {row_count}")
    cells = build_cells(x_lines, y_lines)
    grid = TableGrid(table, cells, row_count, col_count, x_lines, y_lines, warnings)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "table_warped.png"), table)
        _write_grid_overlay(grid, debug_dir / "grid_overlay.png")
    return grid


def scan_document(image, debug_dir: Path | None = None):
    warnings: list[str] = []
    height, width = image.shape[:2]
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in _paper_contours(image):
        area = cv2.contourArea(contour)
        if area < width * height * 0.18:
            continue
        peri = cv2.arcLength(contour, True)
        for epsilon_ratio in (0.015, 0.02, 0.03, 0.04, 0.06):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                points = approx.reshape(4, 2).astype("float32")
                x, y, w, h = cv2.boundingRect(points.astype("int32"))
                if w < width * 0.45 or h < height * 0.45:
                    continue
                rectangularity = area / max(w * h, 1)
                center_bonus = 1.0 - abs((x + w / 2) - width / 2) / width
                score = area / (width * height) + rectangularity * 0.3 + center_bonus * 0.2
                candidates.append((score, points))
                break
    if not candidates:
        warnings.append("paper scan fallback used")
        return image, warnings

    candidates.sort(key=lambda item: item[0], reverse=True)
    warped = warp_perspective(image, _expand_quad(candidates[0][1], image.shape, padding_ratio=0.01))
    if debug_dir:
        cv2.imwrite(str(debug_dir / "paper_warped.png"), warped)
        overlay = image.copy()
        cv2.polylines(overlay, [candidates[0][1].astype("int32")], True, (0, 0, 255), 4)
        cv2.imwrite(str(debug_dir / "paper_contour.png"), overlay)
    return warped, warnings


def _paper_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return sorted(contours, key=cv2.contourArea, reverse=True)

    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 7)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)


def find_and_warp_main_table(image, debug_dir: Path | None = None):
    warnings: list[str] = []
    mask = make_line_mask(image)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "line_mask.png"), mask)
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    connected = cv2.dilate(connected, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    image_area = width * height
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < width * 0.25 or h < height * 0.18 or area < image_area * 0.03:
            continue
        line_pixels = cv2.countNonZero(mask[y : y + h, x : x + w])
        area_score = area / image_area
        width_score = min(w / max(width * 0.55, 1), 1.5)
        height_score = min(h / max(height * 0.35, 1), 1.5)
        line_count_score = min(line_pixels / max(area * 0.08, 1), 1.5)
        right_side_penalty = 0.6 if x > width * 0.45 else 0.0
        too_narrow_penalty = 1.0 if w < width * 0.55 else 0.0
        score = area_score + width_score + height_score + line_count_score - right_side_penalty - too_narrow_penalty
        candidates.append((score, area, x, y, w, h, contour))

    if not candidates:
        warnings.append("table perspective fallback used")
        return image, warnings

    candidates.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    _, _, x, y, w, h, contour = candidates[0]
    support_contours = _supporting_table_contours(mask, x, y, w, h)
    if support_contours:
        points = np.vstack([candidate.reshape(-1, 2) for candidate in support_contours]).astype("float32")
    else:
        points = contour.reshape(-1, 2).astype("float32")

    hull = cv2.convexHull(points.astype("int32"))
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    warped_candidates = []
    if len(approx) == 4:
        warped = warp_perspective(image, _expand_quad(approx.reshape(4, 2).astype("float32"), image.shape))
        warped_candidates.append(("approxPolyDP", *_quick_grid_counts(warped), warped))

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype("float32")
    box_w, box_h = rect[1]
    if min(box_w, box_h) > 1 and max(box_w, box_h) > width * 0.45:
        warped = warp_perspective(image, _expand_quad(box, image.shape))
        warped_candidates.append(("minAreaRect", *_quick_grid_counts(warped), warped))

    sx, sy, sw, sh = cv2.boundingRect(points.astype("int32"))
    pad = max(8, int(min(sw, sh) * 0.02))
    x1 = max(0, sx - pad)
    y1 = max(0, sy - pad)
    x2 = min(width, sx + sw + pad)
    y2 = min(height, sy + sh + pad)
    crop = image[y1:y2, x1:x2]
    if crop.size:
        warped_candidates.append(("boundingRect", *_quick_grid_counts(crop), crop))

    if warped_candidates:
        warped_candidates.sort(key=lambda item: (min(item[1], 18) + min(item[2], 30), item[1] >= 10, item[2] >= 6), reverse=True)
        method, rows, cols, warped = warped_candidates[0]
        if rows >= 8 and cols >= 6:
            if method != "approxPolyDP":
                warnings.append(f"table perspective {method} used")
            return warped, warnings

    warnings.append("table perspective fallback used")
    return crop if crop.size else image, warnings


def _supporting_table_contours(mask, x: int, y: int, w: int, h: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    support = []
    x1, x2 = x, x + w
    y_limit_top = max(0, y - int(h * 0.6))
    y_limit_bottom = min(mask.shape[0], y + h + int(h * 0.12))
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        cx2 = cx + cw
        overlap = max(0, min(x2, cx2) - max(x1, cx))
        overlap_ratio = overlap / max(min(w, cw), 1)
        is_wide_line = cw >= w * 0.35 and overlap_ratio >= 0.45
        is_inside_band = y_limit_top <= cy + ch and cy <= y_limit_bottom
        is_table_vertical = ch >= h * 0.18 and overlap_ratio >= 0.15
        if is_inside_band and (is_wide_line or is_table_vertical):
            support.append(contour)
    return support


def _expand_quad(points: np.ndarray, image_shape, padding_ratio: float = 0.015) -> np.ndarray:
    height, width = image_shape[:2]
    center = points.mean(axis=0)
    expanded = points.copy().astype("float32")
    for index, point in enumerate(expanded):
        vector = point - center
        expanded[index] = point + vector * padding_ratio
        expanded[index][0] = min(max(expanded[index][0], 0), width - 1)
        expanded[index][1] = min(max(expanded[index][1], 0), height - 1)
    return expanded


def _quick_grid_counts(image) -> tuple[int, int]:
    horizontal, vertical = make_line_masks(image)
    height, width = image.shape[:2]
    best_rows = 0
    best_cols = 0
    for ratio in (0.08, 0.06, 0.04, 0.03, 0.02):
        y_threshold = max(int(width * ratio), 20)
        x_threshold = max(int(height * ratio), 20)
        y_lines = _with_edges(
            merge_positions([int(i) for i, value in enumerate(np.count_nonzero(horizontal, axis=1)) if value >= y_threshold], max(3, height // 300)),
            height,
        )
        x_lines = _with_edges(
            merge_positions([int(i) for i, value in enumerate(np.count_nonzero(vertical, axis=0)) if value >= x_threshold], max(3, width // 300)),
            width,
        )
        best_rows = max(best_rows, len(y_lines) - 1)
        best_cols = max(best_cols, len(x_lines) - 1)
    return best_rows, best_cols


def detect_grid_lines(table_image, debug_dir: Path | None = None) -> tuple[list[int], list[int]]:
    horizontal, vertical = make_line_masks(table_image)
    height, width = table_image.shape[:2]
    y_projection = np.count_nonzero(horizontal, axis=1)
    x_projection = np.count_nonzero(vertical, axis=0)
    x_lines: list[int] = []
    y_lines: list[int] = []
    for ratio in (0.08, 0.06, 0.04, 0.03, 0.02):
        y_threshold = max(int(width * ratio), 20)
        x_threshold = max(int(height * ratio), 20)
        candidate_y = _with_edges(
            merge_positions([int(i) for i, value in enumerate(y_projection) if value >= y_threshold], max(3, height // 300)),
            height,
        )
        candidate_x = _with_edges(
            merge_positions([int(i) for i, value in enumerate(x_projection) if value >= x_threshold], max(3, width // 300)),
            width,
        )
        x_lines, y_lines = candidate_x, candidate_y
        if len(candidate_x) >= 7 and len(candidate_y) >= 11:
            break
    if len(x_lines) > 31:
        x_lines = _merge_until_count(x_lines, 31)
    if len(y_lines) > 80:
        y_lines = _merge_until_count(y_lines, 80)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "horizontal_mask.png"), horizontal)
        cv2.imwrite(str(debug_dir / "vertical_mask.png"), vertical)
    return x_lines, y_lines


def merge_positions(values: list[int], tolerance: int) -> list[int]:
    if not values:
        return []
    groups: list[list[int]] = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(sum(group) / len(group))) for group in groups]


def build_cells(x_lines: list[int], y_lines: list[int]) -> list[CellBox]:
    cells: list[CellBox] = []
    for row in range(len(y_lines) - 1):
        for col in range(len(x_lines) - 1):
            x1, x2 = x_lines[col], x_lines[col + 1]
            y1, y2 = y_lines[row], y_lines[row + 1]
            if x2 > x1 and y2 > y1:
                cells.append(CellBox(row, col, x1, y1, x2 - x1, y2 - y1))
    return cells


def _with_edges(lines: list[int], max_value: int) -> list[int]:
    lines = [value for value in lines if 0 <= value <= max_value - 1]
    if not lines or lines[0] > max_value * 0.04:
        lines.insert(0, 0)
    if lines[-1] < max_value * 0.96:
        lines.append(max_value - 1)
    return sorted(set(lines))


def _merge_until_count(lines: list[int], max_count: int) -> list[int]:
    lines = list(lines)
    while len(lines) > max_count:
        gaps = [(lines[index + 1] - lines[index], index) for index in range(len(lines) - 1)]
        _, index = min(gaps)
        merged = int(round((lines[index] + lines[index + 1]) / 2))
        lines[index : index + 2] = [merged]
    return lines


def _regular_lines(max_value: int, count: int) -> list[int]:
    if count < 1:
        return [0, max_value - 1]
    return [int(round(value)) for value in np.linspace(0, max_value - 1, count + 1)]


def _write_grid_overlay(grid: TableGrid, path: Path) -> None:
    image = grid.image.copy()
    for x in grid.x_lines:
        cv2.line(image, (x, 0), (x, image.shape[0] - 1), (0, 255, 0), 1)
    for y in grid.y_lines:
        cv2.line(image, (0, y), (image.shape[1] - 1, y), (0, 255, 0), 1)
    for row in range(grid.row_count):
        y = grid.y_lines[row] + 14
        cv2.putText(image, str(row), (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for col in range(grid.col_count):
        x = grid.x_lines[col] + 4
        cv2.putText(image, str(col), (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
