from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass

import cv2

from .cell_ocr import CellOcrEngine, parse_kart_no, parse_lap_time
from .grid_parser import looks_like_avg, looks_like_lap_nr
from .image_preprocess import resize_max_side
from .lap_row_detector import OcrWord
from .models import KartResult
from .parser import ParsedSheet, _find_date, _find_heat, _find_time


@dataclass(frozen=True)
class DirectParseDebug:
    words: list[OcrWord]
    lap_word: OcrWord | None
    avg_word: OcrWord | None
    kart_words: list[OcrWord]
    avg_values: list[OcrWord]
    lap_chains: list[list[OcrWord]]
    avg_matches: list[OcrWord | None]


def recognize_lap_sheet_direct(
    image_path: Path,
    max_side: int = 1600,
    debug_dir: Path | None = None,
    ocr_lang: str = "ch",
    ocr_device: str = "cpu",
    ocr_enable_mkldnn: bool = False,
    ocr_cpu_threads: int = 1,
) -> ParsedSheet:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = resize_max_side(image, max_side)
    image_dir = _debug_image_dir(debug_dir) if debug_dir else None
    if image_dir:
        cv2.imwrite(str(image_dir / "direct_original_resized.png"), image)

    words = CellOcrEngine(
        lang=ocr_lang,
        device=ocr_device,
        enable_mkldnn=ocr_enable_mkldnn,
        cpu_threads=ocr_cpu_threads,
    ).recognize_image_words(image)
    parsed, debug = _parse_direct_ocr_words_with_debug(words)
    _write_direct_debug(image, debug, parsed, debug_dir)
    return parsed


def parse_direct_ocr_words(words: list[OcrWord]) -> ParsedSheet:
    parsed, _ = _parse_direct_ocr_words_with_debug(words)
    return parsed


def _parse_direct_ocr_words_with_debug(words: list[OcrWord]) -> tuple[ParsedSheet, DirectParseDebug]:
    sheet = ParsedSheet()
    all_text = " ".join(word.text for word in sorted(words, key=lambda item: (item.y, item.x)))
    sheet.date = _find_date(all_text)
    sheet.printed_time = _find_time(all_text)
    sheet.heat = _find_heat(all_text)

    lap_word = _first_word(words, looks_like_lap_nr)
    if lap_word is None:
        sheet.warnings.append("Lap/Nr cell not detected")
        sheet.raw_debug_summary = {"mode": "direct_paddleocr", "word_count": len(words)}
        return sheet, DirectParseDebug(words, None, None, [], [], [], [])

    avg_word = _first_word(words, looks_like_avg)
    kart_words = _right_chain(words, lap_word, parse_kart_no)
    avg_values = _right_chain(words, avg_word, parse_lap_time) if avg_word else []
    lap_bottom_y = _lap_bottom_y(avg_word, avg_values)

    karts: list[KartResult] = []
    lap_chains: list[list[OcrWord]] = []
    avg_matches: list[OcrWord | None] = []
    for position, kart_word in enumerate(kart_words, start=1):
        lap_words = _lap_chain_below(words, kart_word, lap_bottom_y)
        laps = [lap for word in lap_words if (lap := parse_lap_time(word.text)) is not None]
        lap_chains.append(lap_words)
        if len(laps) < 2:
            avg_matches.append(None)
            continue
        avg_match = _matching_avg_word(avg_values, position)
        avg_matches.append(avg_match)
        printed_avg = parse_lap_time(avg_match.text) if avg_match else None
        karts.append(
            KartResult(
                kart_no=parse_kart_no(kart_word.text),
                position=position,
                best_lap=min(laps),
                avg_lap=printed_avg,
                laps=laps,
            )
        )

    sheet.karts = karts
    if not karts:
        sheet.warnings.append("No kart lap columns detected")
    sheet.raw_debug_summary = {
        "mode": "direct_paddleocr",
        "word_count": len(words),
        "lap_nr": _word_debug(lap_word),
        "avg": _word_debug(avg_word) if avg_word else None,
        "kart_count": len(kart_words),
        "kart_words": [_word_debug(word) for word in kart_words],
        "avg_values": [_word_debug(word) for word in avg_values],
        "lap_chains": [[_word_debug(word) for word in chain] for chain in lap_chains],
        "avg_matches": [_word_debug(word) if word else None for word in avg_matches],
    }
    debug = DirectParseDebug(words, lap_word, avg_word, kart_words, avg_values, lap_chains, avg_matches)
    return sheet, debug


def _first_word(words: list[OcrWord], predicate) -> OcrWord | None:
    matches = [word for word in words if predicate(word.text)]
    return sorted(matches, key=lambda item: (item.y, item.x))[0] if matches else None


def _right_chain(words: list[OcrWord], anchor: OcrWord | None, parser) -> list[OcrWord]:
    if anchor is None:
        return []
    selected: list[OcrWord] = []
    current = anchor
    used: set[int] = set()
    while True:
        candidate = _best_right_by_y_overlap(words, current, used, parser)
        if candidate is None:
            return selected
        selected.append(candidate)
        used.add(id(candidate))
        current = candidate


def _best_right_by_y_overlap(words: list[OcrWord], anchor: OcrWord, used: set[int], parser) -> OcrWord | None:
    candidates = []
    for word in words:
        if id(word) in used or parser(word.text) is None:
            continue
        x_delta = word.center_x - anchor.center_x
        if x_delta <= 0:
            continue
        y_overlap = _overlap(anchor.y, anchor.y + anchor.h, word.y, word.y + word.h)
        if y_overlap <= 0:
            continue
        candidates.append((y_overlap / x_delta, y_overlap, x_delta, word))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], -item[2]))[3]


def _lap_chain_below(words: list[OcrWord], kart_word: OcrWord, bottom_y: float | None) -> list[OcrWord]:
    laps: list[OcrWord] = []
    current = kart_word
    used: set[int] = set()
    while True:
        candidate = _best_below_by_x_overlap(words, current, used, bottom_y)
        if candidate is None:
            return laps
        used.add(id(candidate))
        lap = parse_lap_time(candidate.text)
        if lap is None:
            return laps
        laps.append(candidate)
        current = candidate


def _best_below_by_x_overlap(words: list[OcrWord], anchor: OcrWord, used: set[int], bottom_y: float | None) -> OcrWord | None:
    candidates = []
    for word in words:
        if id(word) in used or word.center_y <= anchor.center_y:
            continue
        if bottom_y is not None and word.y + word.h > bottom_y:
            continue
        if parse_lap_time(word.text) is None:
            continue
        x_overlap = _overlap(anchor.x, anchor.x + anchor.w, word.x, word.x + word.w)
        if x_overlap <= 0:
            continue
        y_delta = word.center_y - anchor.center_y
        if y_delta <= 0:
            continue
        candidates.append((x_overlap / y_delta, x_overlap, y_delta, word))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], -item[2]))[3]


def _matching_avg_word(avg_values: list[OcrWord], position: int) -> OcrWord | None:
    if position - 1 < len(avg_values):
        return avg_values[position - 1]
    return None


def _lap_bottom_y(avg_word: OcrWord | None, avg_values: list[OcrWord]) -> float | None:
    candidates = []
    if avg_word is not None:
        candidates.append(avg_word.y)
    candidates.extend(word.y for word in avg_values)
    return min(candidates) if candidates else None


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _word_debug(word: OcrWord) -> dict[str, object]:
    return {
        "text": word.text,
        "confidence": word.confidence,
        "x": round(word.x, 2),
        "y": round(word.y, 2),
        "w": round(word.w, 2),
        "h": round(word.h, 2),
    }


def _write_direct_debug(image, debug: DirectParseDebug, parsed: ParsedSheet, debug_dir: Path | None) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    image_dir = _debug_image_dir(debug_dir)
    _write_words_json(debug.words, debug_dir / "direct_ocr_words.json")
    _write_parsed_json(parsed, debug_dir / "direct_parsed.json")
    overlay = image.copy()
    for word in debug.words:
        _draw_word(overlay, word, (0, 128, 255), word.text[:16])
    cv2.imwrite(str(image_dir / "direct_paddleocr_overlay.png"), overlay)

    selected = image.copy()
    for word in debug.words:
        _draw_word(selected, word, (90, 90, 90), None)
    if debug.lap_word:
        _draw_word(selected, debug.lap_word, (0, 0, 255), "Lap/Nr")
    if debug.avg_word:
        _draw_word(selected, debug.avg_word, (0, 128, 255), "Avg")
    for index, word in enumerate(debug.kart_words, start=1):
        _draw_word(selected, word, (255, 0, 0), f"K{index}:{word.text}")
        if debug.lap_word:
            _draw_link(selected, debug.lap_word, word, (255, 0, 0))
    for index, word in enumerate(debug.avg_values, start=1):
        _draw_word(selected, word, (0, 180, 180), f"A{index}:{word.text}")
        if debug.avg_word:
            _draw_link(selected, debug.avg_word, word, (0, 180, 180))
    cv2.imwrite(str(image_dir / "direct_selected_boxes.png"), selected)

    chains = image.copy()
    for index, (kart_word, chain) in enumerate(zip(debug.kart_words, debug.lap_chains, strict=False), start=1):
        _draw_word(chains, kart_word, (255, 0, 0), f"K{index}:{kart_word.text}")
        previous = kart_word
        for lap_index, lap_word in enumerate(chain, start=1):
            _draw_word(chains, lap_word, (0, 255, 0), f"{index}-{lap_index}:{lap_word.text}")
            _draw_link(chains, previous, lap_word, (0, 255, 0))
            previous = lap_word
    cv2.imwrite(str(image_dir / "direct_lap_chains.png"), chains)

    avgs = image.copy()
    for index, (kart_word, avg_word) in enumerate(zip(debug.kart_words, debug.avg_matches, strict=False), start=1):
        _draw_word(avgs, kart_word, (255, 0, 0), f"K{index}:{kart_word.text}")
        if avg_word:
            _draw_word(avgs, avg_word, (0, 180, 180), f"Avg{index}:{avg_word.text}")
            _draw_link(avgs, kart_word, avg_word, (0, 180, 180))
    cv2.imwrite(str(image_dir / "direct_avg_links.png"), avgs)


def _draw_word(image, word: OcrWord, color: tuple[int, int, int], label: str | None) -> None:
    x1, y1 = int(round(word.x)), int(round(word.y))
    x2, y2 = int(round(word.x + word.w)), int(round(word.y + word.h))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(image, label[:24], (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _draw_link(image, source: OcrWord, target: OcrWord, color: tuple[int, int, int]) -> None:
    start = (int(round(source.center_x)), int(round(source.center_y)))
    end = (int(round(target.center_x)), int(round(target.center_y)))
    cv2.line(image, start, end, color, 2)
    cv2.circle(image, start, 3, color, -1)
    cv2.circle(image, end, 3, color, -1)


def _write_words_json(words: list[OcrWord], path: Path) -> None:
    payload = [_word_debug(word) for word in sorted(words, key=lambda item: (item.y, item.x))]
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_parsed_json(parsed: ParsedSheet, path: Path) -> None:
    payload = {
        "date": parsed.date,
        "printed_time": parsed.printed_time,
        "heat": parsed.heat,
        "warnings": parsed.warnings,
        "karts": [kart.to_dict() for kart in parsed.karts],
        "raw_debug_summary": parsed.raw_debug_summary,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _debug_image_dir(debug_dir: Path) -> Path:
    image_dir = debug_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir
