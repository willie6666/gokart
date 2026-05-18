from __future__ import annotations

import csv
import json
import shutil
import threading
from pathlib import Path
from typing import Any

from .models import KartResult, SessionRecord, UserRecord, now_iso


class JsonStore:
    def __init__(self, path: Path) -> None:
        self.root = path if path.suffix == "" else path.parent
        self.settings_path = self.root / "settings.json"
        self._lock = threading.RLock()
        self._settings = self._load_settings()
        self._users = self._settings.setdefault("users", {})

    def _load_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return {"version": 2, "next_session_id": self._next_id_from_record_dirs(), "settings": {}, "users": {}}
        with self.settings_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        data.setdefault("version", 1)
        data.setdefault("next_session_id", self._next_id_from_record_dirs())
        data.setdefault("settings", {})
        data.setdefault("users", {})
        return data

    def _next_id_from_record_dirs(self) -> int:
        if not self.root.exists():
            return 1
        ids = [int(path.name) for path in self.root.iterdir() if path.is_dir() and path.name.isdigit()]
        return max(ids, default=0) + 1

    def save(self) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp_path = self.settings_path.with_suffix(f"{self.settings_path.suffix}.tmp")
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(self._settings, file, ensure_ascii=False, indent=2, sort_keys=True)
                file.write("\n")
            tmp_path.replace(self.settings_path)

    def next_session_id(self) -> str:
        with self._lock:
            session_id = str(self._settings["next_session_id"])
            self._settings["next_session_id"] += 1
            self.save()
            return session_id

    def record_dir(self, session_id: str) -> Path:
        return self.root / str(session_id)

    def images_dir(self, session_id: str) -> Path:
        return self.record_dir(session_id) / "images"

    def upsert_user(self, user_id: int, display_name: str) -> UserRecord:
        with self._lock:
            users = self._users
            key = str(user_id)
            timestamp = now_iso()
            if key in users:
                user = UserRecord.from_dict(users[key])
                user.display_name = display_name
                user.updated_at = timestamp
            else:
                user = UserRecord(user_id, display_name, timestamp, timestamp)
            users[key] = user.to_dict()
            self.save()
            return user

    def add_session(self, session: SessionRecord) -> None:
        with self._lock:
            self._write_session_files(session)

    def update_session(self, session: SessionRecord) -> None:
        self.add_session(session)

    def get_session(self, session_id: str) -> SessionRecord | None:
        path = self.record_dir(str(session_id)) / "session.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        data["raw_ocr"] = self._read_raw_ocr(str(session_id))
        data["karts"] = self._read_karts(str(session_id))
        return SessionRecord.from_dict(data)

    def all_sessions(self) -> list[SessionRecord]:
        if not self.root.exists():
            return []
        sessions = []
        for path in sorted((path for path in self.root.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda item: int(item.name)):
            session = self.get_session(path.name)
            if session is not None:
                sessions.append(session)
        return sessions

    def sessions_with_result_messages(self) -> list[SessionRecord]:
        return [session for session in self.all_sessions() if session.result_message_id]

    def get_record_channel_id(self) -> int | None:
        return self._get_channel_setting("record_channel_id")

    def set_record_channel_id(self, channel_id: int | None) -> None:
        self._set_channel_setting("record_channel_id", channel_id)

    def get_debug_channel_id(self) -> int | None:
        return self._get_channel_setting("debug_channel_id")

    def set_debug_channel_id(self, channel_id: int | None) -> None:
        self._set_channel_setting("debug_channel_id", channel_id)

    def _get_channel_setting(self, key: str) -> int | None:
        value = self._settings["settings"].get(key)
        return int(value) if value is not None else None

    def _set_channel_setting(self, key: str, channel_id: int | None) -> None:
        with self._lock:
            if channel_id is None:
                self._settings["settings"].pop(key, None)
            else:
                self._settings["settings"][key] = int(channel_id)
            self.save()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            path = self.record_dir(str(session_id))
            if not path.exists():
                raise ValueError(f"Session {session_id} not found")
            shutil.rmtree(path)

    def claim_kart(self, session_id: str, kart_no: int, user_id: int, display_name: str) -> SessionRecord:
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        kart = session.find_kart(kart_no)
        if kart is None:
            raise ValueError(f"Kart {kart_no} not found in session {session_id}")
        if kart.position is None:
            raise ValueError(f"Kart {kart_no} has no claimable position")
        return self.claim_position(session_id, kart.position, user_id, display_name)

    def claim_position(self, session_id: str, position: int, user_id: int, display_name: str) -> SessionRecord:
        with self._lock:
            self.upsert_user(user_id, display_name)
            session = self.get_session(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")
            kart = next((candidate for candidate in session.karts if candidate.position == position), None)
            if kart is None:
                raise ValueError(f"Position {position} not found in session {session_id}")
            kart_label = f"kart {kart.kart_no}" if kart.kart_no is not None else f"position {position}"
            if kart.claimed_by_user_id is not None and kart.claimed_by_user_id != user_id:
                raise ValueError(f"{kart_label} is already claimed by {_driver_label(kart)}")
            for other in session.karts:
                if other.claimed_by_user_id == user_id and other.position != position:
                    other_label = f"kart {other.kart_no}" if other.kart_no is not None else f"position {other.position}"
                    raise ValueError(
                        f"You already claimed {other_label} in this session. "
                        "Use /unclaim first if you need to change it."
                    )
            kart.claimed_by_user_id = user_id
            kart.claimed_by_name = display_name
            kart.claimed_at = now_iso()
            self.update_session(session)
            return session

    def _write_session_files(self, session: SessionRecord) -> None:
        record_dir = self.record_dir(session.id)
        record_dir.mkdir(parents=True, exist_ok=True)
        with (record_dir / "session.json").open("w", encoding="utf-8") as file:
            json.dump(_session_metadata(session), file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        with (record_dir / "raw_ocr.json").open("w", encoding="utf-8") as file:
            json.dump(session.raw_ocr, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        self._write_karts_csv(session, record_dir / "karts.csv")
        self._write_laps_csv(session, record_dir / "laps.csv")

    def _write_karts_csv(self, session: SessionRecord, path: Path) -> None:
        fields = [
            "position",
            "kart_no",
            "best_lap",
            "avg_lap",
            "claimed_by_user_id",
            "claimed_by_name",
            "claimed_at",
        ]
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for kart in session.karts:
                writer.writerow(
                    {
                        "position": kart.position,
                        "kart_no": kart.kart_no,
                        "best_lap": kart.best_lap,
                        "avg_lap": kart.avg_lap,
                        "claimed_by_user_id": kart.claimed_by_user_id,
                        "claimed_by_name": kart.claimed_by_name,
                        "claimed_at": kart.claimed_at,
                    }
                )

    def _write_laps_csv(self, session: SessionRecord, path: Path) -> None:
        fields = ["position", "lap_index", "lap_time"]
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for kart in session.karts:
                for index, lap in enumerate(kart.laps, start=1):
                    writer.writerow({"position": kart.position, "lap_index": index, "lap_time": lap})

    def _read_raw_ocr(self, session_id: str) -> dict[str, Any]:
        path = self.record_dir(session_id) / "raw_ocr.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as file:
            return dict(json.load(file))

    def _read_karts(self, session_id: str) -> list[dict[str, Any]]:
        record_dir = self.record_dir(session_id)
        karts_path = record_dir / "karts.csv"
        if not karts_path.exists():
            return []
        laps = self._read_laps(record_dir / "laps.csv")
        karts = []
        with karts_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                position = _optional_int(row.get("position"))
                karts.append(
                    {
                        "position": position,
                        "kart_no": _optional_int(row.get("kart_no")),
                        "best_lap": _optional_float(row.get("best_lap")),
                        "avg_lap": _optional_float(row.get("avg_lap")),
                        "claimed_by_user_id": _optional_int(row.get("claimed_by_user_id")),
                        "claimed_by_name": row.get("claimed_by_name") or None,
                        "claimed_at": row.get("claimed_at") or None,
                        "laps": laps.get(position, []),
                    }
                )
        return karts

    def _read_laps(self, path: Path) -> dict[int | None, list[float]]:
        if not path.exists():
            return {}
        laps: dict[int | None, list[tuple[int, float]]] = {}
        with path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                position = _optional_int(row.get("position"))
                lap_index = _optional_int(row.get("lap_index")) or 0
                lap_time = _optional_float(row.get("lap_time"))
                if lap_time is None:
                    continue
                laps.setdefault(position, []).append((lap_index, lap_time))
        return {position: [lap for _, lap in sorted(values)] for position, values in laps.items()}

    def unclaim_user(self, session_id: str, user_id: int) -> SessionRecord:
        with self._lock:
            session = self.get_session(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")
            changed = False
            for kart in session.karts:
                if kart.claimed_by_user_id == user_id:
                    kart.claimed_by_user_id = None
                    kart.claimed_by_name = None
                    kart.claimed_at = None
                    changed = True
            if not changed:
                raise ValueError("You have not claimed a kart in this session")
            self.update_session(session)
            return session

    def unclaim_position(self, session_id: str, position: int, user_id: int) -> SessionRecord:
        with self._lock:
            session = self.get_session(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")
            kart = next((candidate for candidate in session.karts if candidate.position == position), None)
            if kart is None:
                raise ValueError(f"Position {position} not found in session {session_id}")
            if kart.claimed_by_user_id != user_id:
                label = f"kart {kart.kart_no}" if kart.kart_no is not None else f"position {position}"
                if kart.claimed_by_user_id is None:
                    raise ValueError(f"{label} is not claimed by you")
                raise ValueError(f"{label} is already claimed by {_driver_label(kart)}")
            kart.claimed_by_user_id = None
            kart.claimed_by_name = None
            kart.claimed_at = None
            self.update_session(session)
            return session
    def fix_kart(
        self, session_id: str, kart_no: int, best_lap: float | None, laps: list[float] | None, position: int | None = None
    ) -> SessionRecord:
        with self._lock:
            session = self.get_session(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")
            kart = session.find_kart(kart_no)
            if kart is None and position is not None:
                for candidate in session.karts:
                    if candidate.position == position:
                        kart = candidate
                        kart.kart_no = kart_no
                        break
            if kart is None:
                kart = KartResult(kart_no=kart_no, position=len(session.karts) + 1)
                session.karts.append(kart)
            if laps is not None:
                kart.laps = laps
                kart.best_lap = min(laps) if laps else None
                kart.avg_lap = round(sum(laps) / len(laps), 3) if laps else None
            if best_lap is not None:
                kart.best_lap = best_lap
            session.karts.sort(key=lambda item: (item.position is None, item.position or 999, item.kart_no))
            self.update_session(session)
            return session


def _driver_label(kart: KartResult) -> str:
    if kart.claimed_by_user_id is not None:
        return f"<@{kart.claimed_by_user_id}>"
    return kart.claimed_by_name or "unknown"


def _session_metadata(session: SessionRecord) -> dict[str, Any]:
    data = session.to_dict()
    data.pop("raw_ocr", None)
    data.pop("karts", None)
    return data


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
