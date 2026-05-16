from __future__ import annotations

from .models import KartResult, SessionRecord


def format_session(session: SessionRecord) -> str:
    title_parts = [f"紀錄 #{session.id}"]
    if session.date:
        title_parts.append(session.date)
    if session.printed_time:
        title_parts.append(session.printed_time)

    lines = [" | ".join(title_parts)]
    if session.ocr_confidence is not None:
        lines.append(f"OCR 信心：{session.ocr_confidence:.2f}")
    if session.warnings:
        lines.append("警告：" + "；".join(session.warnings[:4]))

    lines.append("")
    if not session.karts:
        lines.append("沒有辨識到車號。請使用 `/fix` 手動新增。")
        return "\n".join(lines)

    lines.append("辨識結果：")
    for kart in session.karts:
        lines.append(format_kart_line(kart))
    lines.append("")
    lines.append("按下方車號按鈕認領。辨識錯誤可用 `/fix` 修正。")
    return "\n".join(lines)


def format_kart_line(kart: KartResult) -> str:
    kart_label = f"車號 {kart.kart_no}" if kart.kart_no is not None else "車號未知"
    best = f"{kart.best_lap:.2f}s" if kart.best_lap is not None else "未知"
    laps = f"{len(kart.laps)} 圈" if kart.laps else "圈數未知"
    claimed = f"，已認領：{kart.claimed_by_name}" if kart.claimed_by_name else ""
    return f"{kart_label}：最佳 {best}，{laps}{claimed}"


def format_laps(session: SessionRecord, kart_no: int | None = None) -> str:
    karts = session.karts
    if kart_no is not None:
        kart = session.find_kart(kart_no)
        if kart is None:
            return f"紀錄 #{session.id} 找不到車號 {kart_no}。"
        karts = [kart]

    heading = _session_label(session)
    lines = [f"完整圈速 | {heading}"]
    if not karts:
        lines.append("這場沒有車號紀錄。")
        return "\n".join(lines)

    for kart in karts:
        lines.append("")
        lines.append(format_kart_line(kart))
        if not kart.laps:
            lines.append("沒有完整圈速資料。")
            continue
        lines.append(_format_lap_table(kart.laps))
    return "\n".join(lines)


def format_user_records(user_id: int, sessions: list[SessionRecord]) -> str:
    claimed: list[tuple[SessionRecord, KartResult]] = []
    for session in sessions:
        for kart in session.karts:
            if kart.claimed_by_user_id == user_id:
                claimed.append((session, kart))

    if not claimed:
        return "你目前沒有認領過任何紀錄。"

    claimed.sort(key=lambda item: item[1].best_lap if item[1].best_lap is not None else 999)
    best_session, best_kart = claimed[0]
    lines = [f"你的紀錄共 {len(claimed)} 場。"]
    lines.append(f"個人最佳：{best_kart.best_lap:.2f}s，車號 {best_kart.kart_no}，紀錄 #{best_session.id}")
    lines.append("")
    for session, kart in claimed[:10]:
        label = _session_label(session)
        best = f"{kart.best_lap:.2f}s" if kart.best_lap is not None else "未知"
        kart_label = f"車號 {kart.kart_no}" if kart.kart_no is not None else "車號未知"
        lines.append(f"#{session.id} {label} {kart_label}：{best}")
    return "\n".join(lines)


def format_leaderboard(sessions: list[SessionRecord], limit: int = 10) -> str:
    rows: list[tuple[SessionRecord, KartResult]] = []
    for session in sessions:
        for kart in session.karts:
            if kart.best_lap is not None:
                rows.append((session, kart))
    rows.sort(key=lambda item: item[1].best_lap or 999)

    if not rows:
        return "目前沒有可排名的紀錄。"

    lines = ["卡丁車排行榜"]
    lines.append("```text")
    lines.append(f"{'#':>2} {'圈速':>7}  {'車手':<14} {'車號':>4}  {'日期/時間':<18} {'紀錄':>5}")
    lines.append("-- -------  -------------- ----  ------------------ -----")
    for index, (session, kart) in enumerate(rows[:limit], 1):
        owner = kart.claimed_by_name or "未認領"
        label = _session_label(session)
        kart_no = str(kart.kart_no) if kart.kart_no is not None else "?"
        lines.append(f"{index:>2} {kart.best_lap:>6.2f}s  {_clip(owner, 14):<14} {kart_no:>4}  {_clip(label, 18):<18} #{session.id:>4}")
    lines.append("```")
    return "\n".join(lines)


def format_record_channel(channel_id: int | None) -> str:
    if channel_id is None:
        return "目前尚未設定紀錄圖片頻道。設定前 bot 不會自動辨識任何圖片。"
    return f"目前紀錄圖片頻道：<#{channel_id}>"


def _session_label(session: SessionRecord) -> str:
    parts = []
    if session.date:
        parts.append(session.date)
    if session.printed_time:
        parts.append(session.printed_time)
    return " ".join(parts) if parts else "未知日期時間"


def _format_lap_table(laps: list[float]) -> str:
    rows = [f"{index + 1:02d}:{lap:05.2f}s" for index, lap in enumerate(laps)]
    lines = []
    for index in range(0, len(rows), 4):
        lines.append("  ".join(rows[index : index + 4]))
    return "```text\n" + "\n".join(lines) + "\n```"


def _clip(value: str, length: int) -> str:
    if len(value) <= length:
        return value
    return value[: max(length - 1, 0)] + "…"
