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

    assert "20.84s" in output
    assert "19.65s" in output
    assert "01:20.84s" not in output


def test_leaderboard_uses_each_claimed_driver_best_once() -> None:
    first = session()
    first.find_kart(12).claimed_by_user_id = 99
    first.karts.append(KartResult(kart_no=5, position=3, best_lap=18.5, claimed_by_name="unclaimed"))
    second = SessionRecord(
        id="2",
        channel_id=10,
        source_message_id=21,
        image_url="https://example.com/image2.jpg",
        author_user_id=1,
        author_name="author",
        created_at=now_iso(),
        date="2026/5/17",
        printed_time="05:00:00",
        karts=[KartResult(kart_no=12, position=1, best_lap=20.0, claimed_by_user_id=99, claimed_by_name="driver")],
    )

    output = format_leaderboard([first, second])

    assert "1. 19.65s｜<@99>｜2026/5/16 04:18:42" in output
    assert "20.00s" not in output
    assert "18.50s" not in output
    assert "Heat 12" not in output


def test_profile_display_includes_mvp_stats() -> None:
    record = session()
    record.find_kart(12).claimed_by_user_id = 99
    output = format_user_records(99, [record])

    assert "總場次：1" in output
    assert "總圈數：2" in output
    assert "個人最佳：19.65s" in output
    assert "平均圈速：20.24s" in output


def test_profile_display_handles_claim_without_best_lap() -> None:
    record = session()
    kart = record.find_kart(12)
    kart.claimed_by_user_id = 99
    kart.best_lap = None
    kart.laps = []

    output = format_user_records(99, [record])

    assert "總場次：1" in output
    assert "個人最佳：未知" in output


def test_myrecords_display_recent_claims() -> None:
    record = session()
    record.find_kart(12).claimed_by_user_id = 99
    output = format_myrecords(99, [record])

    assert "最近 1 場紀錄" in output
    assert "車號 12" in output
