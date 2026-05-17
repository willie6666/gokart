from __future__ import annotations

from pathlib import Path
from statistics import median

import cv2
import numpy as np


DEFAULT_MAX_OCR_SIDE = 1400


def preprocess_image(source: Path, destination: Path, max_side: int = DEFAULT_MAX_OCR_SIDE, rectify_table: bool = True) -> Path:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Cannot read image: {source}")

    if rectify_table:
        image = _rectify_table(image)
    image = _resize_for_ocr(image, max_side)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), binary)
    return destination


def resize_max_side(image, max_side: int):
    height, width = image.shape[:2]
    current = max(width, height)
    if current <= max_side:
        return image
    scale = max_side / current
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def make_ocr_image(image):
    """Build a contrast-enhanced OCR image without destroying decimal points."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)


def make_line_masks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    horizontal = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1)),
    )
    vertical = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35)),
    )
    return horizontal, vertical


def make_line_mask(image):
    horizontal, vertical = make_line_masks(image)
    return cv2.bitwise_or(horizontal, vertical)


def order_points(points):
    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def warp_perspective(image, points):
    rect = order_points(points)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(int(width_a), int(width_b), 1)
    max_height = max(int(height_a), int(height_b), 1)
    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def _rectify_table(image):
    height, width = image.shape[:2]
    scale = 1200 / max(height, width)
    small = cv2.resize(image, (int(width * scale), int(height * scale)))
    line_mask = _table_line_mask(small)
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    image_area = small.shape[0] * small.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > image_area * 0.02 and w > small.shape[1] * 0.3 and h > small.shape[0] * 0.2:
            candidates.append((area, x, y, w, h))
    if not candidates:
        return image

    _, x, y, w, h = max(candidates)
    padding = 40
    inv_scale = 1 / scale
    x1 = max(0, int((x - padding) * inv_scale))
    y1 = max(0, int((y - padding) * inv_scale))
    x2 = min(width, int((x + w + padding) * inv_scale))
    y2 = min(height, int((y + h + padding) * inv_scale))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image

    angle = _estimate_horizontal_angle(crop)
    if abs(angle) < 0.2:
        return crop

    crop_height, crop_width = crop.shape[:2]
    matrix = cv2.getRotationMatrix2D((crop_width / 2, crop_height / 2), angle, 1.0)
    return cv2.warpAffine(crop, matrix, (crop_width, crop_height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def _table_line_mask(image):
    return make_line_mask(image)


def _estimate_horizontal_angle(image) -> float:
    import math

    height, width = image.shape[:2]
    scale = 1000 / max(height, width)
    small = cv2.resize(image, (int(width * scale), int(height * scale)))
    line_mask = _table_line_mask(small)
    lines = cv2.HoughLinesP(line_mask, 1, np.pi / 180, threshold=80, minLineLength=120, maxLineGap=20)
    if lines is None:
        return 0.0

    angles = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(value) for value in line]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 120:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -12 <= angle <= 12:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(median(angles))


def _resize_for_ocr(image, max_side: int):
    return resize_max_side(image, max_side)

