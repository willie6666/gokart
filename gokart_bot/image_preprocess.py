from __future__ import annotations

from pathlib import Path
from statistics import median


DEFAULT_MAX_OCR_SIDE = 2200


def preprocess_image(source: Path, destination: Path, max_side: int = DEFAULT_MAX_OCR_SIDE, rectify_table: bool = True) -> Path:
    import cv2

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


def _rectify_table(image):
    import cv2
    import numpy as np

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
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    horizontal = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1)))
    vertical = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35)))
    return horizontal | vertical


def _estimate_horizontal_angle(image) -> float:
    import cv2
    import math
    import numpy as np

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
    height, width = image.shape[:2]
    current_max_side = max(width, height)
    if current_max_side <= max_side:
        return image

    import cv2

    scale = max_side / current_max_side
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    return width, height
