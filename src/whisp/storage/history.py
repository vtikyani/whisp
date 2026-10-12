"""Local dictation history and usage stats, kept in SQLite.

The database lives at ``AppPaths.history_db``. It runs in WAL mode with one
connection guarded by a lock, so every method is safe to call from any
thread. Deleted rows are overwritten on disk (``secure_delete``) and the WAL
is truncated after deletions, so cleared text doesn't linger in the file.

Search uses an FTS5 index on ``final_text`` when the SQLite build has FTS5,
and falls back to simple substring matching when it doesn't.

When ``privacy.history_enabled`` is off, every dictation is still recorded
for stats, but with empty ``raw_text`` and ``final_text``.

Never log entry text. Log ids and counts instead.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from whisp.storage.paths import display_path
from whisp.storage.settings import SettingsStore

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
"""Bump when the schema changes, and add the upgrade steps to ``_MIGRATIONS``."""

TYPING_WPM = 40
"""Typing speed that "time saved" is measured against."""

STATS_OUTCOMES = ("inserted", "copied")
"""Outcomes that count towards stats: the text reached the user."""

PRUNE_INTERVAL_MS = 24 * 60 * 60 * 1000
MAX_PAGE = 1000

_COLUMNS = (
    "id",
    "created_at",
    "kind",
    "app_name",
    "process_name",
    "category",
    "raw_text",
    "final_text",
    "word_count",
    "audio_ms",
    "speech_ms",
    "latency_ms",
    "model_id",
    "polished",
    "outcome",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM entries"

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE entries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   REAL    NOT NULL,
            day          TEXT    NOT NULL,
            kind         TEXT    NOT NULL,
            app_name     TEXT    NOT NULL DEFAULT '',
            process_name TEXT    NOT NULL DEFAULT '',
            category     TEXT    NOT NULL DEFAULT 'other',
            raw_text     TEXT    NOT NULL DEFAULT '',
            final_text   TEXT    NOT NULL DEFAULT '',
            word_count   INTEGER NOT NULL DEFAULT 0,
            audio_ms     INTEGER NOT NULL DEFAULT 0,
            speech_ms    INTEGER NOT NULL DEFAULT 0,
            latency_ms   INTEGER NOT NULL DEFAULT 0,
            model_id     TEXT    NOT NULL DEFAULT '',
            polished     INTEGER NOT NULL DEFAULT 0,
            outcome      TEXT    NOT NULL
        )
        """,
        "CREATE INDEX entries_created_at ON entries(created_at)",
        "CREATE INDEX entries_app ON entries(app_name, created_at)",
        "CREATE INDEX entries_day ON entries(day)",
    ),
}
"""Schema upgrades: version -> statements that take the previous version to it."""

_FTS_CLAUSE = "id IN (SELECT rowid FROM entries_fts WHERE entries_fts MATCH ?)"
_FTS_TABLE = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5("
    "final_text, content='entries', content_rowid='id', tokenize='unicode61 remove_diacritics 2')"
)
_FTS_TRIGGERS = {
    "entries_fts_ai": (
        "CREATE TRIGGER entries_fts_ai AFTER INSERT ON entries BEGIN "
        "INSERT INTO entries_fts(rowid, final_text) VALUES (new.id, new.final_text); END"
    ),
    "entries_fts_ad": (
        "CREATE TRIGGER entries_fts_ad AFTER DELETE ON entries BEGIN "
        "INSERT INTO entries_fts(entries_fts, rowid, final_text) VALUES ('delete', old.id, old.final_text); END"
    ),
    "entries_fts_au": (
        "CREATE TRIGGER entries_fts_au AFTER UPDATE OF final_text ON entries BEGIN "
        "INSERT INTO entries_fts(entries_fts, rowid, final_text) VALUES ('delete', old.id, old.final_text); "
        "INSERT INTO entries_fts(rowid, final_text) VALUES (new.id, new.final_text); END"
    ),
}

_SQLITE_CORRUPT = 11
_SQLITE_NOTADB = 26

# Chinese and Japanese don't put spaces between words, so each character counts as one word.
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_WORD_RE = re.compile(rf"[{_CJK}]|[^\s{_CJK}]+")


def count_words(text: str) -> int:
    """Count words the way people expect: whitespace-separated words, ignoring lone punctuation.

    Chinese and Japanese characters count as one word each.
    """
    return sum(1 for token in _WORD_RE.findall(text) if any(ch.isalnum() for ch in token))


@dataclass(frozen=True)
class HistoryEntry:
    """One finished dictation or command.

    ``word_count`` is the number of words in ``final_text`` for dictations, and
    in the spoken instruction (``raw_text``) for commands, so words-per-minute
    stays meaningful for both.
    """

    id: int
    created_at: float
    """``time.time()`` when the recording started."""
    kind: str
    """``"dictation"`` or ``"command"``."""
    app_name: str
    process_name: str
    category: str
    raw_text: str = field(repr=False)
    """What the speech model heard. Empty when history text is turned off."""
    final_text: str = field(repr=False)
    """What was inserted. Empty when history text is turned off."""
    word_count: int
    audio_ms: int
    speech_ms: int
    latency_ms: int
    model_id: str
    polished: bool
    outcome: str
    """An ``Outcome`` value: ``"inserted"``, ``"copied"`` or ``"failed"``."""


class _CorruptDatabase(sqlite3.DatabaseError):
    """The file opened, but SQLite's integrity check failed."""


class HistoryStore(QObject):
    """Thread-safe SQLite history with search and stats. See the module docstring."""

    added = Signal(object)
    """HistoryEntry, as stored (text blanked when history text is off). Emitted on the thread that called ``add``."""
    changed = Signal()
    """Entries were deleted, cleared or pruned."""
    cleared = Signal()
    """The user cleared all history: anything else holding past text (paste-last) should forget it too."""

    def __init__(self, db_path: Path, settings: SettingsStore, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._path = Path(db_path)
        self._settings = settings
        self._lock = threading.RLock()
        self._closed = False
        self._fts = False
        self._persistent = True
        self._maintenance_timer: QTimer | None = None
        self._conn = self._open()
        settings.changed.connect(self._on_settings_changed)

    # -- public API (any thread) ---------------------------------------------------

    @property
    def persistent(self) -> bool:
        """False when the database file couldn't be opened and history lives in memory for this session."""
        return self._persistent

    @property
    def search_uses_fts(self) -> bool:
        """True when search runs on the FTS5 index rather than substring matching."""
        return self._fts

    def add(self, entry: HistoryEntry) -> HistoryEntry:
        """Store ``entry`` (its ``id`` is ignored) and return it with the new id."""
        keep_text = self._settings.current.privacy.history_enabled
        raw_text = entry.raw_text if keep_text else ""
        final_text = entry.final_text if keep_text else ""
        values = (
            float(entry.created_at),
            _local_day(entry.created_at),
            entry.kind,
            entry.app_name,
            entry.process_name,
            entry.category,
            raw_text,
            final_text,
            int(entry.word_count),
            int(entry.audio_ms),
            int(entry.speech_ms),
            int(entry.latency_ms),
            entry.model_id,
            1 if entry.polished else 0,
            entry.outcome,
        )
        with self._lock:
            conn = self._connection()
            with self._transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO entries (created_at, day, kind, app_name, process_name, category, raw_text,"
                    " final_text, word_count, audio_ms, speech_ms, latency_ms, model_id, polished, outcome)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
                new_id = int(cursor.lastrowid or 0)
        stored = dataclasses.replace(entry, id=new_id, raw_text=raw_text, final_text=final_text)
        self.added.emit(stored)
        return stored

    def list(self, *, query: str = "", app: str = "", limit: int = 50, offset: int = 0) -> list[HistoryEntry]:
        """Newest first. ``query`` matches words in the inserted text (the last word as a prefix)."""
        limit = max(1, min(int(limit), MAX_PAGE))
        offset = max(0, int(offset))
        query = query.strip()
        with self._lock:
            conn = self._connection()
            match = _fts_query(query) if query and self._fts else None
            if match is not None:
                try:
                    return self._select(conn, _FTS_CLAUSE, [match], app, limit, offset)
                except sqlite3.OperationalError:
                    log.warning("Full-text search failed; falling back to simple matching", exc_info=True)
            clause, params = _like_clause(query) if query else ("", [])
            return self._select(conn, clause, params, app, limit, offset)

    def get(self, entry_id: int) -> HistoryEntry | None:
        with self._lock:
            row = self._connection().execute(f"{_SELECT} WHERE id = ?", (int(entry_id),)).fetchone()
        return _entry(row) if row is not None else None

    def delete(self, entry_id: int) -> None:
        with self._lock:
            conn = self._connection()
            with self._transaction(conn):
                removed = conn.execute("DELETE FROM entries WHERE id = ?", (int(entry_id),)).rowcount
            if removed:
                self._checkpoint(conn)
        if removed:
            self.changed.emit()

    def clear(self) -> None:
        """Delete every entry, and any damaged copy set aside earlier. The text is overwritten on disk, not just
        unlinked."""
        with self._lock:
            conn = self._connection()
            with self._transaction(conn):
                removed = conn.execute("DELETE FROM entries").rowcount
            self._checkpoint(conn)
        copies = self._remove_damaged_copies(written_before=None)
        log.info("History cleared (%d entries, %d damaged copies)", removed, copies)
        self.cleared.emit()
        self.changed.emit()

    def prune(self, *, now: float | None = None) -> int:
        """Delete entries older than ``privacy.history_retention_days`` (0 keeps everything). Returns the count.

        A damaged copy of the database (see ``_set_aside_damaged_file``) goes once everything in it is that old.
        """
        days = self._settings.current.privacy.history_retention_days
        if days <= 0:
            return 0
        cutoff = (time.time() if now is None else now) - days * 86_400
        with self._lock:
            conn = self._connection()
            with self._transaction(conn):
                removed = conn.execute("DELETE FROM entries WHERE created_at < ?", (cutoff,)).rowcount
            if removed:
                self._checkpoint(conn)
        copies = self._remove_damaged_copies(written_before=cutoff)
        if copies:
            log.info("Removed %d damaged history copies older than %d days", copies, days)
        if removed:
            log.info("Removed %d history entries older than %d days", removed, days)
            self.changed.emit()
        return removed

    def last_text(self) -> str | None:
        """The most recent non-empty inserted text, or None."""
        sql = "SELECT final_text FROM entries WHERE final_text != '' ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._lock:
            row = self._connection().execute(sql).fetchone()
        return str(row[0]) if row is not None else None

    def apps(self) -> list[str]:
        """Names of the apps that appear in history, alphabetically."""
        with self._lock:
            rows = (
                self._connection()
                .execute("SELECT DISTINCT app_name FROM entries WHERE app_name != '' ORDER BY app_name COLLATE NOCASE")
                .fetchall()
            )
        return [str(row[0]) for row in rows]

    def stats(self, *, now: float | None = None) -> dict[str, Any]:
        """Usage stats. Days are local calendar days; weeks start on Monday.

        Keys: ``total_words``, ``total_dictations``, ``words_today``, ``words_this_week``,
        ``avg_wpm`` (words per speaking minute), ``time_saved_minutes`` (vs typing at
        40 wpm, minus speaking time), ``streak_days`` (consecutive days up to today, or
        up to yesterday if you haven't dictated yet today), ``best_streak_days``,
        ``top_apps`` (``[{app, words}]``, top 5) and ``daily`` (the last 30 days, oldest
        first, ``[{date: "YYYY-MM-DD", words}]``).
        """
        today = _local_date(time.time() if now is None else now)
        counted = ", ".join("?" for _ in STATS_OUTCOMES)
        with self._lock:
            conn = self._connection()
            dictations, words, speech_ms = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(word_count), 0), COALESCE(SUM(speech_ms), 0)"
                f" FROM entries WHERE outcome IN ({counted})",
                STATS_OUTCOMES,
            ).fetchone()
            day_rows = conn.execute(
                f"SELECT day, COALESCE(SUM(word_count), 0) FROM entries WHERE outcome IN ({counted}) GROUP BY day",
                STATS_OUTCOMES,
            ).fetchall()
            app_rows = conn.execute(
                "SELECT app_name, SUM(word_count) AS words FROM entries"
                f" WHERE outcome IN ({counted}) AND app_name != ''"
                " GROUP BY app_name ORDER BY words DESC, app_name COLLATE NOCASE LIMIT 5",
                STATS_OUTCOMES,
            ).fetchall()

        words_by_day: dict[date, int] = {}
        for day_text, day_words in day_rows:
            try:
                words_by_day[date.fromisoformat(day_text)] = int(day_words)
            except (TypeError, ValueError):
                log.warning("Skipping a history day with an unreadable date")
        week_start = today - timedelta(days=today.weekday())
        speaking_minutes = int(speech_ms) / 60_000
        total_words = int(words)
        streak, best = _streaks({d for d in words_by_day if d <= today}, today)
        daily = []
        for back in range(29, -1, -1):
            day = today - timedelta(days=back)
            daily.append({"date": day.isoformat(), "words": words_by_day.get(day, 0)})
        return {
            "total_words": total_words,
            "total_dictations": int(dictations),
            "words_today": words_by_day.get(today, 0),
            "words_this_week": sum(w for d, w in words_by_day.items() if week_start <= d <= today),
            "avg_wpm": round(total_words / speaking_minutes) if speaking_minutes > 0 else 0,
            "time_saved_minutes": round(max(0.0, total_words / TYPING_WPM - speaking_minutes), 1),
            "streak_days": streak,
            "best_streak_days": best,
            "top_apps": [{"app": str(app), "words": int(app_words)} for app, app_words in app_rows],
            "daily": daily,
        }

    def start_maintenance(self) -> None:
        """Prune old entries now, then once a day. Call once, on the GUI thread, after the QApplication exists."""
        if self._maintenance_timer is None:
            timer = QTimer(self)
            timer.setInterval(PRUNE_INTERVAL_MS)
            timer.timeout.connect(self._prune_in_background)
            timer.start()
            self._maintenance_timer = timer
        self._prune_in_background()

    def close(self) -> None:
        """Close the database. Further calls raise ``RuntimeError``."""
        if self._maintenance_timer is not None:
            self._maintenance_timer.stop()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.close()
            except sqlite3.Error:
                log.exception("Error while closing the history database")

    # -- internals -------------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("history is closed")
        return self._conn

    @staticmethod
    @contextlib.contextmanager
    def _transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            # Inside the try: a COMMIT that fails (disk full, I/O error) can leave the transaction open,
            # and then every later BEGIN on this shared connection would fail too.
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _checkpoint(conn: sqlite3.Connection) -> None:
        # Move deleted pages out of the WAL file so removed text doesn't stay there.
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        except sqlite3.Error:
            log.warning("Couldn't checkpoint the history database", exc_info=True)

    def _select(
        self, conn: sqlite3.Connection, clause: str, params: Sequence[object], app: str, limit: int, offset: int
    ) -> list[HistoryEntry]:
        where = [clause] if clause else []
        args = list(params)
        if app:
            where.append("app_name = ?")
            args.append(app)
        sql = _SELECT
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [_entry(row) for row in conn.execute(sql, args).fetchall()]

    @Slot()
    def _prune_in_background(self) -> None:
        threading.Thread(target=self._prune_quietly, name="whisp-history-prune", daemon=True).start()

    def _prune_quietly(self) -> None:
        try:
            self.prune()
        except Exception:
            if not self._closed:
                log.exception("Couldn't remove old history entries")

    @Slot(object)
    def _on_settings_changed(self, sections: set[str]) -> None:
        if "privacy" in sections and not self._closed:
            # A shorter retention period applies right away.
            self._prune_in_background()

    # -- opening, migrations, FTS ----------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return self._prepare(str(self._path))
        except sqlite3.DatabaseError as exc:
            if not _is_corruption(exc):
                return self._open_in_memory()
            backup = self._set_aside_damaged_file()
            log.warning(
                "The history database was damaged (%s); moved it to %s and started a new one",
                exc,
                display_path(backup),
            )
        except OSError:
            return self._open_in_memory()
        try:
            return self._prepare(str(self._path))
        except (sqlite3.DatabaseError, OSError):
            return self._open_in_memory()

    def _open_in_memory(self) -> sqlite3.Connection:
        log.exception(
            "Couldn't open the history database at %s; history won't be saved this session", display_path(self._path)
        )
        self._persistent = False
        return self._prepare(":memory:")

    def _prepare(self, target: str) -> sqlite3.Connection:
        conn = sqlite3.connect(target, timeout=5.0, check_same_thread=False, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
            if target != ":memory:" and mode != "wal":
                log.warning("History database is using journal mode %r instead of WAL", mode)
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA secure_delete=ON")
            check = conn.execute("PRAGMA quick_check").fetchone()[0]
            if check != "ok":
                raise _CorruptDatabase(f"integrity check failed: {check}")
            self._migrate(conn)
            self._fts = self._prepare_fts(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        current = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if current > SCHEMA_VERSION:
            log.warning(
                "History database was written by a newer Whisp (schema %d, this one knows %d); using it as is",
                current,
                SCHEMA_VERSION,
            )
            return
        for version in range(current + 1, SCHEMA_VERSION + 1):
            with self._transaction(conn):
                for statement in _MIGRATIONS[version]:
                    conn.execute(statement)
                # PRAGMA can't take parameters; ``version`` is our own integer.
                conn.execute(f"PRAGMA user_version = {int(version)}")
            if current:
                log.info("History database upgraded to schema %d", version)

    def _prepare_fts(self, conn: sqlite3.Connection) -> bool:
        if not _fts5_available(conn):
            # Triggers left by a SQLite build with FTS5 would make every insert fail here.
            with self._transaction(conn):
                for name in _FTS_TRIGGERS:
                    conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            log.info("This SQLite build has no FTS5; history search uses simple matching")
            return False
        placeholders = ", ".join("?" for _ in _FTS_TRIGGERS)
        existing = {
            row[0]
            for row in conn.execute(
                f"SELECT name FROM sqlite_master WHERE type = 'trigger' AND name IN ({placeholders})",
                tuple(_FTS_TRIGGERS),
            )
        }
        with self._transaction(conn):
            conn.execute(_FTS_TABLE)
            missing = [name for name in _FTS_TRIGGERS if name not in existing]
            for name in missing:
                conn.execute(_FTS_TRIGGERS[name])
            if missing:
                # A new database, or rows were added while FTS5 wasn't available: index everything.
                conn.execute("INSERT INTO entries_fts(entries_fts) VALUES ('rebuild')")
        try:
            # Remove deleted words from the index right away instead of leaving tombstones (SQLite 3.42+).
            conn.execute("INSERT INTO entries_fts(entries_fts, rank) VALUES ('secure-delete', 1)")
        except sqlite3.OperationalError:
            log.debug("FTS5 secure-delete isn't supported by this SQLite build")
        return True

    def _set_aside_damaged_file(self) -> Path:
        """Move a damaged database aside, so it can still be rescued with SQLite tools.

        The copy holds dictated text, so it follows the history rules: ``prune`` deletes it once everything in it
        is past the retention period, and ``clear`` deletes it at once.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self._path.with_name(f"{self._path.stem}.damaged-{stamp}{self._path.suffix}")
        for suffix in ("", "-wal", "-shm"):
            source = Path(f"{self._path}{suffix}")
            if source.exists():
                try:
                    os.replace(source, Path(f"{backup}{suffix}"))
                except OSError:
                    log.exception("Couldn't move %s aside", source.name)
        return backup

    def _remove_damaged_copies(self, *, written_before: float | None) -> int:
        """Delete damaged copies set aside earlier (each with its -wal/-shm files); None deletes them all.

        With ``written_before``, only copies whose files were all last written before that time go: nothing in
        them can be newer. Returns how many copies were deleted.
        """
        copies: dict[str, list[Path]] = {}
        try:
            for path in self._path.parent.glob(f"{glob.escape(self._path.stem)}.damaged-*"):
                copies.setdefault(path.name.removesuffix("-wal").removesuffix("-shm"), []).append(path)
        except OSError:
            log.warning("Couldn't look for damaged history copies", exc_info=True)
            return 0
        deleted = 0
        for files in copies.values():
            try:
                if written_before is not None:
                    written: list[float] = []
                    for path in files:
                        with contextlib.suppress(FileNotFoundError):  # another prune may be deleting it too
                            written.append(path.stat().st_mtime)
                    if written and max(written) >= written_before:
                        continue
                for path in files:
                    path.unlink(missing_ok=True)
            except OSError:
                log.warning("Couldn't delete a damaged history copy", exc_info=True)
                continue
            deleted += 1
        return deleted


# -- helpers ---------------------------------------------------------------------------


def _entry(row: sqlite3.Row) -> HistoryEntry:
    return HistoryEntry(
        id=int(row["id"]),
        created_at=float(row["created_at"]),
        kind=str(row["kind"]),
        app_name=str(row["app_name"]),
        process_name=str(row["process_name"]),
        category=str(row["category"]),
        raw_text=str(row["raw_text"]),
        final_text=str(row["final_text"]),
        word_count=int(row["word_count"]),
        audio_ms=int(row["audio_ms"]),
        speech_ms=int(row["speech_ms"]),
        latency_ms=int(row["latency_ms"]),
        model_id=str(row["model_id"]),
        polished=bool(row["polished"]),
        outcome=str(row["outcome"]),
    )


def _local_date(timestamp: float) -> date:
    try:
        return datetime.fromtimestamp(timestamp).date()
    except (OverflowError, OSError, ValueError):
        log.warning("Unusable timestamp %r; using today's date", timestamp)
        return date.today()


def _local_day(timestamp: float) -> str:
    return _local_date(timestamp).isoformat()


def _streaks(days: set[date], today: date) -> tuple[int, int]:
    """(current streak, best streak) in days. A streak still counts if today hasn't had a dictation yet."""
    best = run = 0
    previous: date | None = None
    for day in sorted(days):
        run = run + 1 if previous is not None and day - previous == timedelta(days=1) else 1
        best = max(best, run)
        previous = day
    anchor = today if today in days else today - timedelta(days=1)
    current = 0
    while anchor in days:
        current += 1
        anchor -= timedelta(days=1)
    return current, best


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.whisp_fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.whisp_fts5_probe")
    except sqlite3.OperationalError:
        return False
    return True


def _fts_query(query: str) -> str | None:
    """Turn what the user typed into a safe FTS5 query: every word must match, the last one as a prefix.

    Each word is quoted, so FTS5 syntax characters (``"*():^-``, ``AND``, ``NEAR``…)
    are searched for as text instead of being parsed. Words made only of
    punctuation produce no tokens and are skipped; returns None if nothing is left.
    """
    terms = query.split()
    parts = []
    for index, term in enumerate(terms):
        if not any(ch.isalnum() for ch in term):
            continue
        part = '"' + term.replace('"', '""') + '"'
        if index == len(terms) - 1:
            part += "*"
        parts.append(part)
    return " ".join(parts) if parts else None


def _like_clause(query: str) -> tuple[str, list[str]]:
    terms = query.split()
    clause = " AND ".join("final_text LIKE ? ESCAPE '\\'" for _ in terms)
    escaped = [term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") for term in terms]
    return f"({clause})", [f"%{term}%" for term in escaped]


def _is_corruption(exc: sqlite3.DatabaseError) -> bool:
    if isinstance(exc, _CorruptDatabase):
        return True
    code = getattr(exc, "sqlite_errorcode", None)
    return code is not None and (code & 0xFF) in (_SQLITE_CORRUPT, _SQLITE_NOTADB)
