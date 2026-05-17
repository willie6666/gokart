from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import KartResult


DATE_RE = re.compile(r"(?:Date\s*[:：]?\s*)?(20\d{2}\s*/\s*\d{1,2}\s*/\s*\d{1,2})", re.IGNORECASE)
COMPACT_DATE_RE = re.compile(r"Date\s*[:：]?\s*(20\d{2})\s*/\s*(\d{2,4})", re.IGNORECASE)
HEAT_RE = re.compile(r"Heat\s*[:：]?\s*(?:Heat\s*)?(\d+)", re.IGNORECASE)
TIME_RE = re.compile(r"(?:Time|Printed)\s*[:：]?\s*(?:上午|下午|AM|PM)?\s*(\d{1,2}:\d{2}(?::\d{2})?)", re.IGNORECASE)
LOOSE_TIME_RE = re.compile(r"(?:Time|Printed).{0,30}?(\d{1,2}:\d{2})(?:\s+(\d{2}))?", re.IGNORECASE)


@dataclass
class ParsedSheet:
    date: str | None = None
    printed_time: str | None = None
    heat: str | None = None
    ocr_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    karts: list[KartResult] = field(default_factory=list)
    raw_debug_summary: dict[str, Any] = field(default_factory=dict)


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
        return f"{year}/{tail[0]}/{tail[2]}"
    if len(tail) == 3:
        return f"{year}/{tail[0]}/{tail[1:]}"
    if len(tail) == 4:
        if int(tail[:2]) > 12 and tail[1] == "1":
            return f"{year}/{tail[0]}/{tail[2:]}"
        return f"{year}/{tail[:2]}/{tail[2:]}"
    return None


def _find_time(text: str) -> str | None:
    best: str | None = None
    match = TIME_RE.search(text)
    if match:
        best = match.group(1)
    match = LOOSE_TIME_RE.search(text)
    if match:
        value, seconds = match.groups()
        candidate = f"{value}:{seconds}" if seconds else value
        if best is None or len(candidate) > len(best):
            best = candidate
    return best


def _find_heat(text: str) -> str | None:
    match = HEAT_RE.search(text)
    return f"Heat {match.group(1)}" if match else None
