from gokart_bot.models import KartResult, SessionRecord, now_iso
from gokart_bot.storage import JsonStore


def test_json_store_claim_and_unclaim(tmp_path) -> None:
    store = JsonStore(tmp_path / "records.json")
    session = SessionRecord(
        id="1",
        channel_id=10,
        source_message_id=20,
        image_url="https://example.com/image.jpg",
        author_user_id=1,
        author_name="author",
        created_at=now_iso(),
        karts=[KartResult(kart_no=12, position=1, best_lap=19.65), KartResult(kart_no=7, position=2, best_lap=20.18)],
    )
    store.add_session(session)

    claimed = store.claim_kart("1", 12, 99, "driver")
    assert claimed.find_kart(12).claimed_by_user_id == 99

    unclaimed = store.unclaim_user("1", 99)
    assert unclaimed.find_kart(12).claimed_by_user_id is None


def test_json_store_rejects_claiming_two_karts_in_same_session(tmp_path) -> None:
    store = JsonStore(tmp_path / "records.json")
    session = SessionRecord(
        id="1",
        channel_id=10,
        source_message_id=20,
        image_url="https://example.com/image.jpg",
        author_user_id=1,
        author_name="author",
        created_at=now_iso(),
        karts=[KartResult(kart_no=12, position=1, best_lap=19.65), KartResult(kart_no=7, position=2, best_lap=20.18)],
    )
    store.add_session(session)

    store.claim_kart("1", 12, 99, "driver")

    try:
        store.claim_kart("1", 7, 99, "driver")
    except ValueError as exc:
        assert "already claimed" in str(exc)
    else:
        raise AssertionError("Expected duplicate same-session claim to fail")

    unchanged = store.get_session("1")
    assert unchanged.find_kart(12).claimed_by_user_id == 99
    assert unchanged.find_kart(7).claimed_by_user_id is None


def test_fix_kart_with_laps(tmp_path) -> None:
    store = JsonStore(tmp_path / "records.json")
    session = SessionRecord(
        id="1",
        channel_id=10,
        source_message_id=20,
        image_url="https://example.com/image.jpg",
        author_user_id=1,
        author_name="author",
        created_at=now_iso(),
    )
    store.add_session(session)

    fixed = store.fix_kart("1", 5, None, [21.0, 19.9, 20.5])
    kart = fixed.find_kart(5)

    assert kart is not None
    assert kart.best_lap == 19.9
    assert kart.avg_lap == 20.467


def test_fix_unknown_kart_by_position(tmp_path) -> None:
    store = JsonStore(tmp_path / "records.json")
    session = SessionRecord(
        id="1",
        channel_id=10,
        source_message_id=20,
        image_url="https://example.com/image.jpg",
        author_user_id=1,
        author_name="author",
        created_at=now_iso(),
        karts=[KartResult(kart_no=None, position=3, best_lap=21.84, laps=[22.0, 21.84])],
    )
    store.add_session(session)

    fixed = store.fix_kart("1", 7, None, None, position=3)
    kart = fixed.find_kart(7)

    assert kart is not None
    assert kart.position == 3
    assert kart.best_lap == 21.84


def test_record_channel_setting(tmp_path) -> None:
    store = JsonStore(tmp_path / "records.json")

    assert store.get_record_channel_id() is None

    store.set_record_channel_id(123)
    assert store.get_record_channel_id() == 123

    reloaded = JsonStore(tmp_path / "records.json")
    assert reloaded.get_record_channel_id() == 123

    reloaded.set_record_channel_id(None)
    assert reloaded.get_record_channel_id() is None
