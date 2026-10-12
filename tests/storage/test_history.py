"""HistoryStore: CRUD, search, pruning, stats, privacy mode and thread safety."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from whisp.storage import history as history_module
from whisp.storage.history import HistoryEntry, HistoryStore, count_words
from whisp.storage.settings import SettingsStore

DAY = 86_400.0


def make_entry(
    text: str = "Hello world",
    *,
    created_at: float | None = None,
    app: str = "Slack",
    process: str = "slack.exe",
    kind: str = "dictation",
    words: int | None = None,
    speech_ms: int = 1_000,
    outcome: str = "inserted",
    raw: str | None = None,
) -> HistoryEntry:
    return HistoryEntry(
        id=0,
        created_at=time.time() if created_at is None else created_at,
        kind=kind,
        app_name=app,
        process_name=process,
        category="messaging",
        raw_text=text if raw is None else raw,
        final_text=text,
        word_count=count_words(text) if words is None else words,
        audio_ms=speech_ms + 500,
        speech_ms=speech_ms,
        latency_ms=120,
        model_id="parakeet-tdt-0.6b-v2",
        polished=False,
        outcome=outcome,
    )


def local_ts(day: date, hour: int = 12) -> float:
    """A timestamp at ``hour`` o'clock local time on ``day``."""
    return datetime(day.year, day.month, day.day, hour).timestamp()


@pytest.fixture
def store(qapp, settings_store: SettingsStore, tmp_path: Path) -> Iterator[HistoryStore]:
    s = HistoryStore(tmp_path / "history.db", settings_store)
    yield s
    s.close()


@pytest.fixture(params=["fts", "like"])
def search_store(
    request: pytest.FixtureRequest, qapp, settings_store: SettingsStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[HistoryStore]:
    """A store that searches with FTS5, or with the substring fallback."""
    if request.param == "like":
        monkeypatch.setattr(history_module, "_fts5_available", lambda _conn: False)
    s = HistoryStore(tmp_path / "history.db", settings_store)
    assert s.search_uses_fts is (request.param == "fts")
    yield s
    s.close()


# -- basics ------------------------------------------------------------------------------


def test_add_returns_entry_with_id_and_get_reads_it_back(store: HistoryStore) -> None:
    added = store.add(make_entry("First dictation", raw="first dictation um"))
    assert added.id > 0
    loaded = store.get(added.id)
    assert loaded == added
    assert loaded.raw_text == "first dictation um"
    assert loaded.polished is False
    assert store.get(added.id + 1000) is None


def test_list_is_newest_first_and_pages(store: HistoryStore) -> None:
    now = time.time()
    ids = [store.add(make_entry(f"entry {i}", created_at=now - 100 + i)).id for i in range(5)]
    listed = [e.id for e in store.list()]
    assert listed == list(reversed(ids))
    assert [e.id for e in store.list(limit=2)] == [ids[4], ids[3]]
    assert [e.id for e in store.list(limit=2, offset=2)] == [ids[2], ids[1]]
    assert store.list(limit=2, offset=10) == []


def test_added_signal_carries_the_stored_entry(store: HistoryStore, qtbot) -> None:
    with qtbot.waitSignal(store.added, timeout=1000) as blocker:
        added = store.add(make_entry("Signal me"))
    assert blocker.args == [added]


def test_delete_and_clear(store: HistoryStore, qtbot) -> None:
    a = store.add(make_entry("keep me"))
    b = store.add(make_entry("delete me"))
    with qtbot.waitSignal(store.changed, timeout=1000):
        store.delete(b.id)
    assert [e.id for e in store.list()] == [a.id]
    with qtbot.assertNotEmitted(store.changed):
        store.delete(9999)
    with qtbot.waitSignal(store.changed, timeout=1000):
        store.clear()
    assert store.list() == []
    assert store.last_text() is None


def test_last_text_skips_entries_without_text(store: HistoryStore, settings_store: SettingsStore) -> None:
    now = time.time()
    store.add(make_entry("older text", created_at=now - 10))
    settings_store.update({"privacy": {"history_enabled": False}})
    store.add(make_entry("private text", created_at=now))
    assert store.last_text() == "older text"


def test_schema_version_and_data_survive_reopen(qapp, settings_store: SettingsStore, tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    first = HistoryStore(path, settings_store)
    added = first.add(make_entry("persist me"))
    first.close()
    second = HistoryStore(path, settings_store)
    try:
        assert second.get(added.id) == added
        assert second.persistent
    finally:
        second.close()
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == history_module.SCHEMA_VERSION
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_closed_store_refuses_work(qapp, settings_store: SettingsStore, tmp_path: Path) -> None:
    s = HistoryStore(tmp_path / "history.db", settings_store)
    s.close()
    s.close()  # idempotent
    with pytest.raises(RuntimeError):
        s.add(make_entry())


# -- search ------------------------------------------------------------------------------

SEARCH_TEXTS = {
    "hello": "Hello world, this is a test.",
    "cpp": 'I love C++ and "quoted" things',
    "cafe": "Meet at the café tomorrow",
    "ops": "Use foo* wildcards AND NOT operators (maybe) NEAR here",
    "percent": "100% sure about under_score names",
    "dont": "Don't stop believing",
    "slash": r"Path is C:\temp\new",
}


@pytest.fixture
def filled(search_store: HistoryStore) -> dict[str, int]:
    now = time.time()
    ids = {}
    for i, (key, text) in enumerate(SEARCH_TEXTS.items()):
        ids[key] = search_store.add(make_entry(text, created_at=now - i)).id
    return ids


def found(store: HistoryStore, query: str) -> set[int]:
    return {e.id for e in store.list(query=query)}


def test_search_matches_words_and_prefixes(search_store: HistoryStore, filled: dict[str, int]) -> None:
    assert found(search_store, "hello") == {filled["hello"]}
    assert found(search_store, "HELLO") == {filled["hello"]}
    assert found(search_store, "hel") == {filled["hello"]}  # the last word matches as a prefix
    assert found(search_store, "world test") == {filled["hello"]}  # every word must match
    assert found(search_store, "world believing") == set()
    assert found(search_store, "   ") == set(filled.values())  # blank query lists everything


@pytest.mark.parametrize(
    ("query", "expected_key"),
    [
        ("c++", "cpp"),
        ('"quoted', "cpp"),
        ('"quoted"', "cpp"),
        ("foo*", "ops"),
        ("(maybe)", "ops"),
        ("AND", "ops"),
        ("NOT", "ops"),
        ("NEAR", "ops"),
        ("100%", "percent"),
        ("under_score", "percent"),
        ("don't", "dont"),
        ("++", "cpp"),
        ("C:\\temp", "slash"),
    ],
)
def test_search_handles_special_characters(
    search_store: HistoryStore, filled: dict[str, int], query: str, expected_key: str
) -> None:
    assert filled[expected_key] in found(search_store, query)


@pytest.mark.parametrize("query", ['"', "*", "(", ")", ":", "-", "^", "NEAR(", "a OR", "%", "_", "\\", "'"])
def test_search_never_errors_on_syntax_characters(
    search_store: HistoryStore, filled: dict[str, int], query: str
) -> None:
    assert isinstance(search_store.list(query=query), list)


def test_like_fallback_treats_wildcards_literally(qapp, settings_store, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(history_module, "_fts5_available", lambda _conn: False)
    s = HistoryStore(tmp_path / "history.db", settings_store)
    try:
        literal = s.add(make_entry("50% off")).id
        s.add(make_entry("500 items"))
        s.add(make_entry("a_b"))
        other = s.add(make_entry("axb")).id
        assert found(s, "50%") == {literal}
        assert other not in found(s, "a_b")
    finally:
        s.close()


def test_fts_search_ignores_accents(qapp, settings_store, tmp_path) -> None:
    s = HistoryStore(tmp_path / "history.db", settings_store)
    try:
        if not s.search_uses_fts:
            pytest.skip("this SQLite build has no FTS5")
        cafe = s.add(make_entry("Meet at the café")).id
        assert found(s, "cafe") == {cafe}
        assert found(s, "café") == {cafe}
    finally:
        s.close()


def test_fts_index_is_rebuilt_after_running_without_fts(qapp, settings_store, tmp_path, monkeypatch) -> None:
    path = tmp_path / "history.db"
    with monkeypatch.context() as patch:
        patch.setattr(history_module, "_fts5_available", lambda _conn: False)
        without_fts = HistoryStore(path, settings_store)
        added = without_fts.add(make_entry("written while search was basic"))
        without_fts.close()
    with_fts = HistoryStore(path, settings_store)
    try:
        if not with_fts.search_uses_fts:
            pytest.skip("this SQLite build has no FTS5")
        assert found(with_fts, "basic") == {added.id}
        later = with_fts.add(make_entry("indexed straight away"))
        assert found(with_fts, "straight") == {later.id}
    finally:
        with_fts.close()


def test_app_filter_and_app_list(store: HistoryStore) -> None:
    slack = store.add(make_entry("in slack", app="Slack"))
    store.add(make_entry("in outlook", app="Outlook", process="outlook.exe"))
    store.add(make_entry("also outlook", app="Outlook", process="outlook.exe"))
    store.add(make_entry("unknown app", app="", process=""))
    assert [e.id for e in store.list(app="Slack")] == [slack.id]
    assert len(store.list(app="Outlook")) == 2
    assert store.list(app="Outlook", query="also")[0].final_text == "also outlook"
    assert store.apps() == ["Outlook", "Slack"]


# -- pruning -----------------------------------------------------------------------------


def test_prune_removes_entries_older_than_retention(store: HistoryStore, settings_store, qtbot) -> None:
    now = time.time()
    settings_store.update({"privacy": {"history_retention_days": 30}})
    old = store.add(make_entry("old", created_at=now - 40 * DAY))
    recent = store.add(make_entry("recent", created_at=now - 10 * DAY))
    with qtbot.waitSignal(store.changed, timeout=1000):
        assert store.prune(now=now) == 1
    assert store.get(old.id) is None
    assert store.get(recent.id) is not None
    assert store.prune(now=now) == 0


def test_prune_keeps_everything_when_retention_is_zero(store: HistoryStore, settings_store) -> None:
    settings_store.update({"privacy": {"history_retention_days": 0}})
    store.add(make_entry("ancient", created_at=time.time() - 3_000 * DAY))
    assert store.prune() == 0
    assert len(store.list()) == 1


def test_shorter_retention_prunes_right_away(store: HistoryStore, settings_store, qtbot) -> None:
    store.add(make_entry("five days old", created_at=time.time() - 5 * DAY))
    keep = store.add(make_entry("fresh"))
    settings_store.update({"privacy": {"history_retention_days": 2}})
    qtbot.waitUntil(lambda: [e.id for e in store.list()] == [keep.id], timeout=3000)


def test_start_maintenance_prunes_in_the_background(store: HistoryStore, settings_store, qtbot) -> None:
    settings_store.update({"privacy": {"history_retention_days": 1}})
    store.add(make_entry("stale", created_at=time.time() - 3 * DAY))
    store.start_maintenance()
    qtbot.waitUntil(lambda: store.list() == [], timeout=3000)


# -- stats -------------------------------------------------------------------------------


def test_stats_on_empty_history(store: HistoryStore) -> None:
    stats = store.stats()
    assert stats["total_words"] == 0
    assert stats["total_dictations"] == 0
    assert stats["avg_wpm"] == 0
    assert stats["time_saved_minutes"] == 0
    assert stats["streak_days"] == 0
    assert stats["best_streak_days"] == 0
    assert stats["top_apps"] == []
    assert len(stats["daily"]) == 30
    assert all(day["words"] == 0 for day in stats["daily"])


def test_stats_totals_speed_and_time_saved(store: HistoryStore) -> None:
    now = time.time()
    store.add(make_entry("x", words=150, speech_ms=60_000, created_at=now - 60))
    store.add(make_entry("y", words=150, speech_ms=60_000, created_at=now - 30, outcome="copied"))
    store.add(make_entry("z", words=999, speech_ms=60_000, created_at=now - 10, outcome="failed"))
    stats = store.stats(now=now)
    assert stats["total_words"] == 300  # failed insertions don't count
    assert stats["total_dictations"] == 2
    assert stats["avg_wpm"] == 150  # 300 words over 2 speaking minutes
    assert stats["time_saved_minutes"] == pytest.approx(300 / 40 - 2)


def test_stats_days_week_and_streaks_use_local_days(store: HistoryStore) -> None:
    today = date(2026, 10, 14)  # a Wednesday
    now = local_ts(today, 18)
    # Active: today, the two days before, then a gap, then a four-day run.
    for back, words in [(0, 10), (1, 20), (2, 30), (5, 1), (6, 1), (7, 1), (8, 1)]:
        store.add(make_entry("w", words=words, created_at=local_ts(today - timedelta(days=back), 9)))
    # Dictations a minute before and a minute after midnight land on different days.
    store.add(make_entry("late", words=4, created_at=local_ts(today - timedelta(days=12), 23) + 59 * 60))
    store.add(make_entry("early", words=6, created_at=local_ts(today - timedelta(days=11), 0) + 60))

    stats = store.stats(now=now)
    assert stats["words_today"] == 10
    assert stats["words_this_week"] == 10 + 20 + 30  # Monday to today
    assert stats["streak_days"] == 3
    assert stats["best_streak_days"] == 4
    daily = {d["date"]: d["words"] for d in stats["daily"]}
    assert len(stats["daily"]) == 30
    assert stats["daily"][-1] == {"date": "2026-10-14", "words": 10}
    assert stats["daily"][0]["date"] == "2026-09-15"
    assert daily["2026-10-02"] == 4
    assert daily["2026-10-03"] == 6


def test_streak_survives_until_the_day_ends(store: HistoryStore) -> None:
    today = date(2026, 10, 14)
    for back in (1, 2):
        store.add(make_entry("w", created_at=local_ts(today - timedelta(days=back))))
    assert store.stats(now=local_ts(today, 8))["streak_days"] == 2
    # Two days without dictating breaks it.
    assert store.stats(now=local_ts(today + timedelta(days=1), 8))["streak_days"] == 0
    assert store.stats(now=local_ts(today + timedelta(days=1), 8))["best_streak_days"] == 2


def test_stats_top_apps(store: HistoryStore) -> None:
    for app, words in [("Slack", 50), ("Outlook", 80), ("Slack", 40), ("Notion", 5), ("", 500)]:
        store.add(make_entry("w", app=app, words=words))
    top = store.stats()["top_apps"]
    assert top == [{"app": "Slack", "words": 90}, {"app": "Outlook", "words": 80}, {"app": "Notion", "words": 5}]


def test_history_disabled_keeps_stats_but_not_text(store: HistoryStore, settings_store, qtbot) -> None:
    settings_store.update({"privacy": {"history_enabled": False}})
    with qtbot.waitSignal(store.added, timeout=1000) as blocker:
        added = store.add(make_entry("secret plans for tomorrow", raw="um secret plans for tomorrow"))
    assert added.final_text == "" and added.raw_text == ""
    assert blocker.args[0].final_text == ""
    stored = store.get(added.id)
    assert stored is not None
    assert stored.final_text == "" and stored.raw_text == ""
    assert stored.word_count == 4
    assert store.stats()["total_words"] == 4
    assert store.list(query="secret") == []


# -- robustness --------------------------------------------------------------------------


def test_thread_safety_smoke(store: HistoryStore) -> None:
    errors: list[BaseException] = []
    start = threading.Barrier(9)

    def writer(n: int) -> None:
        try:
            start.wait()
            for i in range(25):
                store.add(make_entry(f"thread {n} entry {i}"))
        except Exception as exc:  # collected and asserted below
            errors.append(exc)

    def reader() -> None:
        try:
            start.wait()
            for _ in range(25):
                store.list(query="thread")
                store.stats()
                store.apps()
        except Exception as exc:  # collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)] + [threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    assert len(store.list(limit=1000)) == 200
    assert store.stats()["total_dictations"] == 200


class _CommitFailsOnce:
    """The store's connection, except that the next COMMIT fails the way SQLite's does on a full disk."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.fail_next_commit = True

    def execute(self, sql: str, *params: object) -> sqlite3.Cursor:
        if sql == "COMMIT" and self.fail_next_commit:
            self.fail_next_commit = False
            raise sqlite3.OperationalError("database or disk is full")
        return self._conn.execute(sql, *params)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)


def test_a_failed_commit_doesnt_break_later_writes(store: HistoryStore) -> None:
    store._conn = _CommitFailsOnce(store._conn)
    with pytest.raises(sqlite3.OperationalError):
        store.add(make_entry("lost to a full disk"))
    # The failed write was rolled back; the next ones work instead of all failing from now on.
    kept = store.add(make_entry("written once there was space"))
    store.delete(kept.id)
    store.add(make_entry("and another"))
    assert [e.final_text for e in store.list()] == ["and another"]


def test_damaged_database_is_set_aside(qapp, settings_store, tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    path.write_bytes(b"this is not a database" * 100)
    s = HistoryStore(path, settings_store)
    try:
        assert s.persistent
        assert s.add(make_entry("fresh start")).id == 1
        assert any(p.name.startswith("history.damaged-") for p in tmp_path.iterdir())
    finally:
        s.close()


def _damaged_copies(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.startswith("history.damaged-"))


def test_damaged_copies_follow_retention_and_go_when_history_is_cleared(
    qapp, settings_store, tmp_path: Path, qtbot
) -> None:
    path = tmp_path / "history.db"
    path.write_bytes(b"this is not a database" * 100)
    s = HistoryStore(path, settings_store)  # sets the damaged file aside: it still holds dictated text
    try:
        [today] = _damaged_copies(tmp_path)
        old = [tmp_path / "history.damaged-20200101-000000.db", tmp_path / "history.damaged-20200101-000000.db-wal"]
        for file in old:
            file.write_bytes(b"dictations from long ago")
            os.utime(file, (time.time() - 400 * DAY,) * 2)

        settings_store.update({"privacy": {"history_retention_days": 0}})  # keep everything
        s.prune()
        assert len(_damaged_copies(tmp_path)) == 3

        settings_store.update({"privacy": {"history_retention_days": 30}})  # also prunes in the background
        s.prune()
        # Only what's past the retention period goes.
        qtbot.waitUntil(lambda: _damaged_copies(tmp_path) == [today], timeout=3000)

        with qtbot.waitSignal(s.cleared, timeout=1000):
            s.clear()
        assert _damaged_copies(tmp_path) == []
    finally:
        s.close()


def test_unopenable_database_falls_back_to_memory(qapp, settings_store, tmp_path: Path) -> None:
    blocked = tmp_path / "history.db"
    blocked.mkdir()  # a folder where the file should be: SQLite can't open it
    s = HistoryStore(blocked, settings_store)
    try:
        assert not s.persistent
        added = s.add(make_entry("kept in memory"))
        assert s.get(added.id) == added
    finally:
        s.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, world!", 2),
        ("", 0),
        ("   ", 0),
        ("— - …", 0),
        ("e-mail the team", 3),
        ("日本語です", 5),
        ("Use 日本 here", 4),
        ("It's 3:30 pm.", 3),
    ],
)
def test_count_words(text: str, expected: int) -> None:
    assert count_words(text) == expected
