from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .image_preprocess import preprocess_image


@dataclass(frozen=True)
class OcrText:
    text: str
    score: float
    box: list[tuple[float, float]]

    @property
    def x(self) -> float:
        return sum(point[0] for point in self.box) / len(self.box)

    @property
    def y(self) -> float:
        return sum(point[1] for point in self.box) / len(self.box)

    @property
    def width(self) -> float:
        xs = [point[0] for point in self.box]
        return max(xs) - min(xs)

    @property
    def height(self) -> float:
        ys = [point[1] for point in self.box]
        return max(ys) - min(ys)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "score": self.score, "box": self.box}


class OcrEngine:
    def __init__(
        self,
        max_side: int = 2200,
        det_limit_side_len: int = 2200,
        det_model: str = "PP-OCRv5_mobile_det",
        rec_model: str = "PP-OCRv5_mobile_rec",
        cpu_threads: int = 1,
    ) -> None:
        self._ocr = None
        self.max_side = max_side
        self.det_limit_side_len = det_limit_side_len
        self.det_model = det_model
        self.rec_model = rec_model
        self.cpu_threads = cpu_threads

    def _load(self) -> Any:
        if self._ocr is None:
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            try:
                import paddle  # noqa: F401
                from paddleocr import PaddleOCR
            except ModuleNotFoundError as exc:
                if exc.name == "paddle":
                    raise RuntimeError(
                        "PaddleOCR needs PaddlePaddle for its default paddle_static engine. "
                        "Install project dependencies again with: pip install -e '.[dev]'"
                    ) from exc
                raise

            self._ocr = PaddleOCR(
                text_detection_model_name=self.det_model,
                text_recognition_model_name=self.rec_model,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=self.det_limit_side_len,
                text_recognition_batch_size=1,
                device="cpu",
                enable_mkldnn=False,
                cpu_threads=self.cpu_threads,
            )
        return self._ocr

    def recognize(self, image_path: Path) -> list[OcrText]:
        processed = image_path.with_name(f"{image_path.stem}.ocr.png")
        try:
            input_path = preprocess_image(image_path, processed, self.max_side)
        except Exception:
            input_path = image_path

        ocr = self._load()
        try:
            result = ocr.predict(str(input_path))
            return _normalize_result(result)
        finally:
            if input_path == processed:
                processed.unlink(missing_ok=True)


def _normalize_result(result: Any) -> list[OcrText]:
    texts: list[OcrText] = []

    for page in result or []:
        if hasattr(page, "json"):
            payload = page.json
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            texts.extend(_from_v3_payload(payload))
            continue

        if isinstance(page, dict):
            texts.extend(_from_v3_payload(page.get("res", page)))
            continue

        if isinstance(page, list):
            texts.extend(_from_legacy_payload(page))

    return texts


def _from_v3_payload(payload: dict[str, Any] | None) -> list[OcrText]:
    if not payload:
        return []
    rec_texts = payload.get("rec_texts") or []
    rec_scores = payload.get("rec_scores") or []
    rec_polys = payload.get("rec_polys") or payload.get("rec_boxes") or []

    items: list[OcrText] = []
    for index, text in enumerate(rec_texts):
        if text is None:
            continue
        score = float(rec_scores[index]) if index < len(rec_scores) else 0.0
        raw_box = rec_polys[index] if index < len(rec_polys) else []
        box = _normalize_box(raw_box)
        if box:
            items.append(OcrText(str(text).strip(), score, box))
    return items


def _from_legacy_payload(payload: list[Any]) -> list[OcrText]:
    items: list[OcrText] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 2:
            continue
        box = _normalize_box(row[0])
        value = row[1]
        if isinstance(value, (list, tuple)) and value:
            text = str(value[0]).strip()
            score = float(value[1]) if len(value) > 1 else 0.0
            items.append(OcrText(text, score, box))
    return items


def _normalize_box(raw_box: Any) -> list[tuple[float, float]]:
    if raw_box is None:
        return []
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4 and all(isinstance(value, (int, float)) for value in raw_box):
        x1, y1, x2, y2 = [float(value) for value in raw_box]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    box: list[tuple[float, float]] = []
    for point in raw_box:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            box.append((float(point[0]), float(point[1])))
    return box
