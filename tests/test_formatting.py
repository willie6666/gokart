from gokart_bot.formatting import format_laps, format_leaderboard, format_myrecords, format_session, format_user_records
from gokart_bot.models import KartResult, SessionRecord, now_iso


def session() -> SessionRecord:
    return SessionRecord(
        id="1",
        channel_id=10,
        source_message_id=20,
        image_url="https://example.com/image.jpg",
        author_user_id=1,
        author_name="author",
        created_at=now_iso(),
        date="2026/5/16",
        printed_time="04:18:42",
        heat="Heat 12",
        karts=[
            KartResult(kart_no=12, position=1, best_lap=19.65, laps=[20.84, 19.65], claimed_by_name="driver"),
            KartResult(kart_no=7, position=2, best_lap=20.18, laps=[21.62, 20.18]),
        ],
    )


def test_session_display_uses_date_time_without_heat() -> None:
    output = format_session(session())

    assert "2026/5/16" in output
    assert "04:18:42" in output
    assert "Heat 12" not in output


def test_laps_display_includes_all_laps() -> None:
    output = format_laps(session(), 12)

    assert "01:20.84s" in output
    assert "02:19.65s" in output


def test_leaderboard_is_table_like_and_hides_heat() -> None:
    output = format_leaderboard([session()])

    assert "```text" in output
    assert "driver" in output
    assert "Heat 12" not in output


def test_profile_display_includes_mvp_stats() -> None:
    record = session()
    record.find_kart(12).claimed_by_user_id = 99
    output = format_user_records(99, [record])

    assert "總場次：1" in output
    assert "總圈數：2" in output
    assert "個人最佳：19.65s" in output
    assert "平均圈速：20.24s" in output


def test_myrecords_display_recent_claims() -> None:
    record = session()
    record.find_kart(12).claimed_by_user_id = 99
    output = format_myrecords(99, [record])

    assert "最近 1 場紀錄" in output
    assert "車號 12" in output
