from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .models import KartResult, SessionRecord, UserRecord, now_iso


class JsonStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "next_session_id": 1, "settings": {}, "sessions": {}, "users": {}}
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        data.setdefault("version", 1)
        data.setdefault("next_session_id", 1)
        data.setdefault("settings", {})
        data.setdefault("sessions", {})
        data.setdefault("users", {})
        return data

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(self._data, file, ensure_ascii=False, indent=2, sort_keys=True)
                file.write("\n")
            tmp_path.replace(self.path)

    def next_session_id(self) -> str:
        with self._lock:
            session_id = str(self._data["next_session_id"])
            self._data["next_session_id"] += 1
            self.save()
            return session_id

    def upsert_user(self, user_id: int, display_name: str) -> UserRecord:
        with self._lock:
            users = self._data["users"]
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
            self._data["sessions"][session.id] = session.to_dict()
            self.save()

    def update_session(self, session: SessionRecord) -> None:
        self.add_session(session)

    def get_session(self, session_id: str) -> SessionRecord | None:
        data = self._data["sessions"].get(str(session_id))
        if data is None:
            return None
        return SessionRecord.from_dict(data)

    def all_sessions(self) -> list[SessionRecord]:
        return [SessionRecord.from_dict(item) for item in self._data["sessions"].values()]

    def sessions_with_result_messages(self) -> list[SessionRecord]:
        return [session for session in self.all_sessions() if session.result_message_id]

    def get_record_channel_id(self) -> int | None:
        value = self._data["settings"].get("record_channel_id")
        return int(value) if value is not None else None

    def set_record_channel_id(self, channel_id: int | None) -> None:
        with self._lock:
            if channel_id is None:
                self._data["settings"].pop("record_channel_id", None)
            else:
                self._data["settings"]["record_channel_id"] = int(channel_id)
            self.save()

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
                raise ValueError(f"{kart_label} is already claimed by {kart.claimed_by_name}")
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
