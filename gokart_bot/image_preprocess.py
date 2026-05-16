from __future__ import annotations

from pathlib import Path


DEFAULT_MAX_OCR_SIDE = 2200


def preprocess_image(source: Path, destination: Path, max_side: int = DEFAULT_MAX_OCR_SIDE) -> Path:
    import cv2

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Cannot read image: {source}")

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
