from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class KartResult:
    kart_no: int | None
    position: int | None = None
    best_lap: float | None = None
    avg_lap: float | None = None
    laps: list[float] = field(default_factory=list)
    claimed_by_user_id: int | None = None
    claimed_by_name: str | None = None
    claimed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kart_no": self.kart_no,
            "position": self.position,
            "best_lap": self.best_lap,
            "avg_lap": self.avg_lap,
            "laps": self.laps,
            "claimed_by_user_id": self.claimed_by_user_id,
            "claimed_by_name": self.claimed_by_name,
            "claimed_at": self.claimed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KartResult":
        return cls(
            kart_no=int(data["kart_no"]) if data.get("kart_no") is not None else None,
            position=data.get("position"),
            best_lap=data.get("best_lap"),
            avg_lap=data.get("avg_lap"),
            laps=[float(value) for value in data.get("laps", [])],
            claimed_by_user_id=data.get("claimed_by_user_id"),
            claimed_by_name=data.get("claimed_by_name"),
            claimed_at=data.get("claimed_at"),
        )


@dataclass
class SessionRecord:
    id: str
    channel_id: int
    source_message_id: int
    image_url: str
    author_user_id: int
    author_name: str
    created_at: str
    result_message_id: int | None = None
    date: str | None = None
    printed_time: str | None = None
    heat: str | None = None
    ocr_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    raw_ocr: dict[str, Any] = field(default_factory=dict)
    karts: list[KartResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "source_message_id": self.source_message_id,
            "result_message_id": self.result_message_id,
            "image_url": self.image_url,
            "author_user_id": self.author_user_id,
            "author_name": self.author_name,
            "created_at": self.created_at,
            "date": self.date,
            "printed_time": self.printed_time,
            "heat": self.heat,
            "ocr_confidence": self.ocr_confidence,
            "warnings": self.warnings,
            "raw_ocr": self.raw_ocr,
            "karts": [kart.to_dict() for kart in self.karts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        return cls(
            id=str(data["id"]),
            channel_id=int(data["channel_id"]),
            source_message_id=int(data["source_message_id"]),
            result_message_id=data.get("result_message_id"),
            image_url=str(data["image_url"]),
            author_user_id=int(data["author_user_id"]),
            author_name=str(data.get("author_name", "unknown")),
            created_at=str(data["created_at"]),
            date=data.get("date"),
            printed_time=data.get("printed_time"),
            heat=data.get("heat"),
            ocr_confidence=data.get("ocr_confidence"),
            warnings=list(data.get("warnings", [])),
            raw_ocr=dict(data.get("raw_ocr", {})),
            karts=[KartResult.from_dict(item) for item in data.get("karts", [])],
        )

    def find_kart(self, kart_no: int) -> KartResult | None:
        for kart in self.karts:
            if kart.kart_no == kart_no:
                return kart
        return None


@dataclass
class UserRecord:
    discord_user_id: int
    display_name: str
    first_seen_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "discord_user_id": self.discord_user_id,
            "display_name": self.display_name,
            "first_seen_at": self.first_seen_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserRecord":
        return cls(
            discord_user_id=int(data["discord_user_id"]),
            display_name=str(data.get("display_name", "unknown")),
            first_seen_at=str(data["first_seen_at"]),
            updated_at=str(data.get("updated_at", data["first_seen_at"])),
        )
