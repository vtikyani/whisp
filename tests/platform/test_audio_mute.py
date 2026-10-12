"""Muting other apps while dictating.

The unit tests cover the exclusion rules, the crash-safety file, ``AudioMuter``'s bookkeeping and
``MuteService``'s worker thread, using fakes. The ``desktop`` test drives real Windows audio sessions, but only
those of child processes it starts itself (they play digital silence). Every other session on the machine is left
alone, and the test puts every mute back the way it found it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import pytest

from whisp.platform.win32 import audio_mute
from whisp.platform.win32.audio_mute import (
    COMMUNICATION_APPS,
    PENDING_MAX_AGE_S,
    STATE_FILE_NAME,
    AudioMuter,
    CoreAudioSessions,
    MutedSession,
    MuteService,
    MuteStateFile,
    SessionSnapshot,
    normalize_process_name,
    owning_app,
    process_start_time,
    process_table,
    skip_reason,
)
from whisp.storage.paths import app_paths

# Windows process ids are multiples of 4, so the odd ids used below never collide with this test process.
OWN_PID = os.getpid()


# -- fakes ---------------------------------------------------------------------------------


@dataclass(eq=False)
class FakeWatch:
    changed_by_others: bool = False
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass(eq=False)
class FakeSession:
    instance_id: str
    session_id: str = ""
    pid: int = 101
    process_started: int | None = None
    process_name: str = "spotify.exe"
    app_pid: int = -1
    app_name: str = ""
    active: bool = True
    system_sounds: bool = False
    muted: bool = False
    refuses_mute: bool = False
    before_mute: Callable[[FakeSession, bool], None] | None = None
    watches: list[FakeWatch] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session_id = self.session_id or f"speakers|{self.process_name}"
        if self.app_pid == -1:
            self.app_pid = self.pid
        self.app_name = self.app_name or self.process_name

    def set_mute(self, muted: bool) -> bool:
        if self.refuses_mute:
            return False
        if self.before_mute is not None:
            self.before_mute(self, muted)
        self.muted = muted
        return True

    def watch(self) -> FakeWatch:
        watch = FakeWatch()
        self.watches.append(watch)
        return watch


class FakeSource:
    def __init__(self, *sessions: FakeSession, capturing: frozenset[int] = frozenset()) -> None:
        self.sessions = list(sessions)
        self.capturing = capturing
        self.snapshots = 0
        self.broken = False

    @contextlib.contextmanager
    def snapshot(self, *, details: bool) -> Iterator[SessionSnapshot]:
        self.snapshots += 1
        if self.broken:
            raise OSError("the Windows audio service is restarting")
        yield SessionSnapshot(sessions=list(self.sessions), capturing_apps=self.capturing if details else frozenset())


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / STATE_FILE_NAME


def saved_ids(path: Path) -> list[str]:
    return sorted(entry["instance_id"] for entry in json.loads(path.read_text(encoding="utf-8"))["sessions"])


def record(name: str, *, age_s: float = 0.0, siblings: tuple[str, ...] = ()) -> MutedSession:
    return MutedSession(
        instance_id=f"{name}|old",
        session_id=f"speakers|{name}.exe",
        pid=103,
        process_name=f"{name}.exe",
        muted_at=time.time() - age_s,
        siblings=siblings,
    )


# -- exclusion rules -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Spotify", "spotify.exe"),
        ("  OBS64.EXE ", "obs64.exe"),
        (r"C:\Program Files\obs-studio\bin\64bit\obs64.exe", "obs64.exe"),
        ("C:/Tools/Foo.Bar", "foo.bar.exe"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_process_names_are_normalised(text: str, expected: str) -> None:
    assert normalize_process_name(text) == expected


def test_call_apps_are_excluded_and_browsers_are_not() -> None:
    required = {
        "ms-teams.exe",
        "teams.exe",
        "zoom.exe",
        "cpthost.exe",
        "webex.exe",
        "ciscocollabhost.exe",
        "slack.exe",
        "discord.exe",
        "skype.exe",
        "whatsapp.exe",
        "telegram.exe",
        "signal.exe",
        "phoneexperiencehost.exe",
    }
    assert required <= COMMUNICATION_APPS
    assert not {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe"} & COMMUNICATION_APPS
    assert all(name == normalize_process_name(name) for name in COMMUNICATION_APPS)


@pytest.mark.parametrize(
    ("session", "reason"),
    [
        (FakeSession("system", pid=0, process_name="", system_sounds=True), "system sounds"),
        (FakeSession("whisp", pid=OWN_PID, process_name="python.exe"), "Whisp itself"),
        (FakeSession("lingering", process_started=0), "app has quit"),
        (FakeSession("paused", active=False), "not playing"),
        (FakeSession("user-muted", muted=True), "muted already"),
        (FakeSession("exited", process_name=""), "unknown app"),
        (FakeSession("teams", process_name="ms-teams.exe"), "call or meeting app"),
        (
            FakeSession(
                "teams-audio", pid=107, process_name="msedgewebview2.exe", app_pid=105, app_name="ms-teams.exe"
            ),
            "call or meeting app",
        ),
        (FakeSession("meet-tab", pid=203, process_name="chrome.exe", app_pid=201), "using a microphone"),
        (FakeSession("youtube-tab", pid=303, process_name="chrome.exe", app_pid=301), None),
        (FakeSession("music"), None),
    ],
    ids=lambda value: value.instance_id if isinstance(value, FakeSession) else str(value),
)
def test_skip_reason(session: FakeSession, reason: str | None) -> None:
    capturing = frozenset({201})  # a Chrome with a Meet call in one of its tabs
    assert skip_reason(session, own_pid=OWN_PID, excluded=COMMUNICATION_APPS, capturing_apps=capturing) == reason


def test_owning_app_follows_helpers_with_the_same_name() -> None:
    processes = {100: (4, "explorer.exe"), 200: (100, "chrome.exe"), 210: (200, "chrome.exe"), 300: (100, "vlc.exe")}
    assert owning_app(210, processes) == (200, "chrome.exe")
    assert owning_app(300, processes) == (300, "vlc.exe")


def test_owning_app_credits_embedded_browser_helpers_to_their_host() -> None:
    processes = {
        100: (4, "explorer.exe"),
        500: (100, "ms-teams.exe"),
        510: (500, "msedgewebview2.exe"),
        520: (510, "msedgewebview2.exe"),
    }
    assert owning_app(520, processes) == (500, "ms-teams.exe")


def test_owning_app_does_not_credit_an_app_to_the_app_that_launched_it() -> None:
    # A browser opened from a link in Slack is still the browser: its YouTube tab gets muted.
    processes = {100: (4, "explorer.exe"), 600: (100, "slack.exe"), 700: (600, "chrome.exe"), 710: (700, "chrome.exe")}
    assert owning_app(710, processes) == (700, "chrome.exe")


def test_owning_app_copes_with_unknown_and_reused_pids() -> None:
    assert owning_app(42, {}) == (42, "")
    assert owning_app(30, {30: (999, "app.exe")}) == (30, "app.exe")
    cycle = {10: (20, "app.exe"), 20: (10, "app.exe")}  # parent ids reused into a loop
    assert owning_app(10, cycle) in ((10, "app.exe"), (20, "app.exe"))


def test_process_table_names_this_process() -> None:
    _parent, name = process_table()[OWN_PID]
    assert name == Path(sys.executable).name.lower()


def test_process_start_time_tells_running_from_exited() -> None:
    started = process_start_time(OWN_PID)
    assert started is not None and started > 0
    assert process_start_time(OWN_PID) == started
    child = subprocess.Popen([sys.executable, "-c", "pass"], creationflags=subprocess.CREATE_NO_WINDOW)
    child.wait(30)
    assert process_start_time(child.pid) == 0  # exited; our handle keeps the pid from being reused meanwhile
    assert process_start_time(0x7FFFFFF0) == 0  # no such process
    assert process_start_time(0) is None


# -- the crash-safety file ---------------------------------------------------------------


def test_state_file_round_trip(state_path: Path) -> None:
    state = MuteStateFile(state_path)
    records = [record("spotify"), record("vlc", siblings=("vlc|other",))]
    assert state.write(records)
    assert state.read() == records
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["format"] == 1
    assert time.time() - data["saved_at"] < 60
    assert data["sessions"][0] == {
        "instance_id": "spotify|old",
        "session_id": "speakers|spotify.exe",
        "pid": 103,
        "process": "spotify.exe",
        "started": None,
        "muted_at": records[0].muted_at,
        "siblings": [],
    }
    assert not state.tmp_path.exists()


def test_a_failed_write_keeps_the_previous_list(state_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = MuteStateFile(state_path)
    first = [record("spotify")]
    assert state.write(first)

    def disk_full(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(audio_mute.os, "fsync", disk_full)
    assert not state.write([record("vlc")])
    assert state.read() == first
    assert not state.tmp_path.exists()


def test_a_half_written_temporary_file_is_ignored(state_path: Path) -> None:
    # What a crash between writing and renaming leaves behind.
    state = MuteStateFile(state_path)
    first, second = [record("spotify")], [record("vlc")]
    assert state.write(first)
    state.tmp_path.write_text('{"format": 1, "sessions": [{"instance', encoding="utf-8")
    assert state.read() == first
    assert state.write(second)
    assert state.read() == second
    assert not state.tmp_path.exists()


@pytest.mark.parametrize(
    "content",
    [
        b"\xff\xfe\x00not utf-8",
        b"{not json",
        b'{"format": 99, "sessions": []}',
        b'{"format": 1, "sessions": {}}',
        b"[]",
    ],
    ids=["not-utf8", "not-json", "unknown-format", "sessions-not-a-list", "not-an-object"],
)
def test_an_unreadable_file_is_removed(state_path: Path, content: bytes) -> None:
    state_path.write_bytes(content)
    assert MuteStateFile(state_path).read() == []
    assert not state_path.exists()


def test_malformed_entries_are_skipped(state_path: Path) -> None:
    good = record("spotify").to_json()
    entries = [
        good,
        {**good, "instance_id": ""},
        {**good, "instance_id": "a", "pid": "103"},
        {**good, "instance_id": "b", "pid": True},
        {**good, "instance_id": "c", "muted_at": None},
        {**good, "instance_id": "d", "siblings": [1]},
        {**good, "instance_id": "e", "session_id": None},
        {**good, "instance_id": "f", "started": "yesterday"},
        "nonsense",
    ]
    state_path.write_text(json.dumps({"format": 1, "sessions": entries}), encoding="utf-8")
    assert [s.instance_id for s in MuteStateFile(state_path).read()] == ["spotify|old"]


def test_delete_removes_both_files_and_can_repeat(state_path: Path) -> None:
    state = MuteStateFile(state_path)
    assert state.write([record("spotify")])
    state.tmp_path.write_text("partial", encoding="utf-8")
    state.delete()
    state.delete()
    assert not state_path.exists()
    assert not state.tmp_path.exists()
    assert state.read() == []


# -- AudioMuter ----------------------------------------------------------------------------


def test_mutes_only_what_should_be_quiet(state_path: Path) -> None:
    music = FakeSession("spotify", pid=101)
    video = FakeSession("youtube-tab", pid=203, process_name="chrome.exe", app_pid=201)
    others = [
        FakeSession("whisp", pid=OWN_PID, process_name="python.exe"),
        FakeSession("system", pid=0, process_name="", system_sounds=True),
        FakeSession("teams", pid=305, process_name="ms-teams.exe"),
        FakeSession("teams-audio", pid=307, process_name="msedgewebview2.exe", app_pid=305, app_name="ms-teams.exe"),
        FakeSession("meet-tab", pid=409, process_name="firefox.exe", app_pid=401),
        FakeSession("user-muted", pid=503, process_name="vlc.exe", muted=True),
        FakeSession("paused", pid=605, process_name="foobar2000.exe", active=False),
        FakeSession("obs", pid=707, process_name="obs64.exe"),
    ]
    source = FakeSource(music, video, *others, capturing=frozenset({401}))
    muter = AudioMuter(["OBS64"], state_file=state_path, source=source)

    assert muter.mute_others() == 2
    assert music.muted and video.muted
    assert [s.instance_id for s in others if s.muted] == ["user-muted"]
    assert saved_ids(state_path) == ["spotify", "youtube-tab"]
    assert muter.is_muting


def test_the_list_is_saved_before_anything_is_muted(state_path: Path) -> None:
    checked: list[str] = []

    def check_saved(session: FakeSession, muted: bool) -> None:
        if not muted:
            return
        entries = json.loads(state_path.read_text(encoding="utf-8"))["sessions"]
        entry = next(e for e in entries if e["instance_id"] == session.instance_id)
        assert (entry["session_id"], entry["pid"], entry["process"]) == (session.session_id, 101, "spotify.exe")
        assert time.time() - entry["muted_at"] < 60
        checked.append(session.instance_id)

    music = FakeSession("spotify", pid=101, before_mute=check_saved)
    assert AudioMuter(state_file=state_path, source=FakeSource(music)).mute_others() == 1
    assert checked == ["spotify"]


def test_restore_unmutes_only_what_it_muted(state_path: Path) -> None:
    music = FakeSession("spotify", pid=101)
    user_muted = FakeSession("vlc", pid=103, process_name="vlc.exe", muted=True)
    muter = AudioMuter(state_file=state_path, source=FakeSource(music, user_muted))
    assert muter.mute_others() == 1

    assert muter.restore() == 1
    assert not music.muted
    assert user_muted.muted
    assert music.watches[0].closed
    assert not state_path.exists()
    assert not muter.is_muting
    assert muter.restore() == 0


def test_restore_leaves_sessions_someone_else_changed(state_path: Path) -> None:
    music = FakeSession("spotify", pid=101)
    video = FakeSession("youtube-tab", pid=203, process_name="chrome.exe")
    muter = AudioMuter(state_file=state_path, source=FakeSource(music, video))
    assert muter.mute_others() == 2
    # Mid-dictation the user unmuted the music and muted it again, and unmuted the video for good.
    music.watches[0].changed_by_others = True
    video.muted = False
    video.watches[0].changed_by_others = True

    assert muter.restore() == 0
    assert music.muted
    assert not video.muted
    assert not muter.has_pending
    assert not state_path.exists()


def test_restore_without_mute_does_nothing(state_path: Path) -> None:
    source = FakeSource(FakeSession("spotify", pid=101, muted=True))
    muter = AudioMuter(state_file=state_path, source=source)
    assert muter.restore() == 0
    assert source.snapshots == 0
    assert source.sessions[0].muted
    assert not state_path.exists()


def test_muting_again_adds_apps_that_started_playing(state_path: Path) -> None:
    music = FakeSession("spotify", pid=101)
    video = FakeSession("youtube-tab", pid=203, process_name="chrome.exe", active=False)
    muter = AudioMuter(state_file=state_path, source=FakeSource(music, video))
    assert muter.mute_others() == 1
    video.active = True
    assert muter.mute_others() == 1
    assert saved_ids(state_path) == ["spotify", "youtube-tab"]
    assert muter.restore() == 2
    assert not music.muted and not video.muted


def test_a_refused_mute_is_neither_counted_nor_kept(state_path: Path) -> None:
    stubborn = FakeSession("stubborn", pid=101, process_name="game.exe", refuses_mute=True)
    music = FakeSession("spotify", pid=103)
    muter = AudioMuter(state_file=state_path, source=FakeSource(stubborn, music))
    assert muter.mute_others() == 1
    assert saved_ids(state_path) == ["spotify"]
    assert stubborn.watches[0].closed


def test_nothing_is_muted_when_the_list_cannot_be_saved(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    not_a_folder = tmp_path / "not-a-folder"
    not_a_folder.write_text("a file where the data folder should be", encoding="utf-8")
    music = FakeSession("spotify", pid=101)
    muter = AudioMuter(state_file=not_a_folder / STATE_FILE_NAME, source=FakeSource(music))
    assert muter.mute_others() == 0
    assert not music.muted
    assert "couldn't save the list" in caplog.text


def test_an_app_that_quits_while_muted_is_unmuted_when_it_starts_again(state_path: Path) -> None:
    music = FakeSession("spotify|1", session_id="speakers|spotify", pid=101)
    user_muted = FakeSession("spotify|2", session_id="speakers|spotify", pid=103, muted=True)
    source = FakeSource(music, user_muted)
    muter = AudioMuter(state_file=state_path, source=source)
    assert muter.mute_others() == 1

    source.sessions.remove(music)  # Spotify quits mid-dictation; Windows saves it as muted
    assert muter.restore() == 0
    assert muter.has_pending
    assert saved_ids(state_path) == ["spotify|1"]
    assert muter.check_pending() == 0  # not back yet

    relaunched = FakeSession("spotify|3", session_id="speakers|spotify", pid=105, muted=True, active=False)
    source.sessions.append(relaunched)
    assert muter.check_pending() == 1
    assert not relaunched.muted
    assert user_muted.muted  # it existed when we muted, so its mute is the user's
    assert not muter.has_pending
    assert not state_path.exists()


def test_a_quit_apps_lingering_session_is_not_taken_for_the_app(state_path: Path) -> None:
    # Right after an app exits, Windows can still list its session for a moment. Unmuting that dead session
    # wouldn't change the mute Windows saved for the app, so it must count as gone.
    music = FakeSession("spotify|1", session_id="speakers|spotify", pid=101, process_started=1111)
    source = FakeSource(music)
    muter = AudioMuter(state_file=state_path, source=source)
    assert muter.mute_others() == 1
    assert json.loads(state_path.read_text(encoding="utf-8"))["sessions"][0]["started"] == 1111

    music.process_started = 0  # Spotify has quit; its session is still listed, still muted
    assert muter.restore() == 0
    assert muter.has_pending
    assert muter.check_pending() == 0

    relaunched = FakeSession("spotify|2", session_id="speakers|spotify", pid=105, process_started=2222, muted=True)
    source.sessions.append(relaunched)
    assert muter.check_pending() == 1
    assert not relaunched.muted
    assert not muter.has_pending


def test_a_reused_pid_is_a_new_process(state_path: Path) -> None:
    # After a reboot the same app can get the same pid, and so the same instance id, as the session in the file.
    left_behind = MutedSession("spotify|101", "speakers|spotify", 101, "spotify.exe", time.time(), (), 1111)
    MuteStateFile(state_path).write([left_behind])
    reborn = FakeSession("spotify|101", session_id="speakers|spotify", pid=101, process_started=2222, muted=True)
    muter = AudioMuter(state_file=state_path, source=FakeSource(reborn))
    assert muter.restore_leftovers() == 1
    assert not reborn.muted
    assert not state_path.exists()


def test_pending_sessions_wait_until_the_dictation_ends(state_path: Path) -> None:
    gone = FakeSession("vlc|1", session_id="speakers|vlc", pid=101, process_name="vlc.exe")
    source = FakeSource(gone)
    muter = AudioMuter(state_file=state_path, source=source)
    assert muter.mute_others() == 1
    source.sessions.remove(gone)
    assert muter.restore() == 0

    music = FakeSession("spotify", pid=103)
    source.sessions.append(music)
    assert muter.mute_others() == 1
    relaunched = FakeSession("vlc|2", session_id="speakers|vlc", pid=105, process_name="vlc.exe", muted=True)
    source.sessions.append(relaunched)
    assert muter.check_pending() == 0  # mid-dictation: stays quiet for now
    assert relaunched.muted

    assert muter.restore() == 2
    assert not relaunched.muted and not music.muted
    assert not state_path.exists()


def test_restore_leftovers_after_a_crash(state_path: Path) -> None:
    music = FakeSession("spotify", pid=101)
    video = FakeSession("youtube-tab", pid=203, process_name="chrome.exe")
    source = FakeSource(music, video)
    assert AudioMuter(state_file=state_path, source=source).mute_others() == 2
    # Whisp crashes here. Before it starts again, the user unmutes the video themselves.
    video.muted = False

    restarted = AudioMuter(state_file=state_path, source=source)
    assert restarted.restore_leftovers() == 1
    assert not music.muted and not video.muted
    assert not state_path.exists()
    assert restarted.restore_leftovers() == 0


def test_restore_leftovers_handles_apps_that_restarted_or_are_away(state_path: Path) -> None:
    MuteStateFile(state_path).write(
        [record("spotify", age_s=60), record("vlc", age_s=60), record("winamp", age_s=PENDING_MAX_AGE_S + 60)]
    )
    restarted = FakeSession("spotify|new", session_id="speakers|spotify.exe", pid=105, muted=True)
    source = FakeSource(restarted)
    muter = AudioMuter(state_file=state_path, source=source)

    assert muter.restore_leftovers() == 1
    assert not restarted.muted
    assert saved_ids(state_path) == ["vlc|old"]  # VLC is still closed; the 30-day-old entry is dropped

    vlc = FakeSession("vlc|new", session_id="speakers|vlc.exe", pid=107, process_name="vlc.exe", muted=True)
    source.sessions.append(vlc)
    assert muter.check_pending() == 1
    assert not vlc.muted
    assert not state_path.exists()


def test_muting_before_restoring_leftovers_keeps_them(state_path: Path) -> None:
    MuteStateFile(state_path).write([record("vlc")])
    relaunched = FakeSession("vlc|new", session_id="speakers|vlc.exe", pid=105, process_name="vlc.exe", muted=True)
    relaunched.active = False
    music = FakeSession("spotify", pid=101)
    muter = AudioMuter(state_file=state_path, source=FakeSource(relaunched, music))

    assert muter.mute_others() == 1
    assert not relaunched.muted
    assert saved_ids(state_path) == ["spotify"]


def test_a_failed_restore_is_retried(state_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    music = FakeSession("spotify", pid=101)
    source = FakeSource(music)
    muter = AudioMuter(state_file=state_path, source=source)
    assert muter.mute_others() == 1

    source.broken = True
    assert muter.restore() == 0
    assert "Couldn't unmute other apps" in caplog.text
    assert music.muted
    assert music.watches[0].closed
    assert muter.has_pending
    assert saved_ids(state_path) == ["spotify"]

    source.broken = False
    assert muter.check_pending() == 1
    assert not music.muted
    assert not state_path.exists()


def test_a_broken_audio_service_mutes_nothing(state_path: Path) -> None:
    source = FakeSource(FakeSession("spotify", pid=101))
    source.broken = True
    muter = AudioMuter(state_file=state_path, source=source)
    assert muter.mute_others() == 0
    assert not muter.is_muting
    assert not state_path.exists()


def test_the_list_lives_in_the_data_folder(isolated_whisp_home: Path) -> None:
    music = FakeSession("spotify", pid=101)
    muter = AudioMuter(source=FakeSource(music))
    assert muter.mute_others() == 1
    path = app_paths().data_dir / STATE_FILE_NAME
    assert path.is_file()
    assert path.is_relative_to(isolated_whisp_home)
    assert muter.restore() == 1
    assert not path.exists()


def test_logs_name_apps_but_never_session_ids(state_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    secret = r"{0.0.0.00000000}.{guid}|\Device\HarddiskVolume3\Users\someone\AppData\Spotify.exe%b{00}|1%b101"
    music = FakeSession(secret, session_id=secret.rsplit("|", 1)[0], pid=101)
    caplog.set_level(logging.DEBUG, logger=audio_mute.__name__)
    muter = AudioMuter(state_file=state_path, source=FakeSource(music))
    assert muter.mute_others() == 1
    assert muter.restore() == 1
    assert "spotify.exe" in caplog.text
    assert "someone" not in caplog.text
    assert all(r.levelno == logging.DEBUG for r in caplog.records)


# -- MuteService -------------------------------------------------------------------------------


class FakeMuter:
    """Records which methods ran, on which thread; can hold the worker up or fail on purpose."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.changed = threading.Condition()
        self.has_pending = False
        self.exclusions: tuple[str, ...] | None = None
        self.leftovers_gate = threading.Event()
        self.leftovers_gate.set()
        self.restore_gate = threading.Event()
        self.restore_gate.set()
        self.failing: set[str] = set()

    def _record(self, name: str) -> int:
        with self.changed:
            self.calls.append((name, threading.current_thread().name))
            self.changed.notify_all()
        if name in self.failing:
            raise RuntimeError(f"{name} failed on purpose")
        return 1

    def names(self) -> list[str]:
        with self.changed:
            return [name for name, _ in self.calls]

    def wait_for(self, predicate: Callable[[list[str]], bool], timeout: float = 5.0) -> None:
        with self.changed:
            ok = self.changed.wait_for(lambda: predicate([name for name, _ in self.calls]), timeout)
        assert ok, f"timed out; calls so far: {self.names()}"

    def set_extra_exclusions(self, names: Any) -> None:
        self.exclusions = tuple(names)
        self._record("set_extra_exclusions")

    def restore_leftovers(self) -> int:
        self.leftovers_gate.wait(10)
        return self._record("restore_leftovers")

    def mute_others(self) -> int:
        return self._record("mute_others")

    def restore(self) -> int:
        self.restore_gate.wait(10)
        return self._record("restore")

    def check_pending(self) -> int:
        return self._record("check_pending")


def test_the_service_works_on_its_own_thread_and_restores_leftovers_first() -> None:
    muter = FakeMuter()
    service = MuteService(muter)
    try:
        service.mute()
        muter.wait_for(lambda names: "mute_others" in names)
        service.restore()
        muter.wait_for(lambda names: "restore" in names)
    finally:
        service.shutdown(5)
    assert muter.names() == ["restore_leftovers", "mute_others", "restore", "restore"]  # the last: on shutdown
    assert {thread for _, thread in muter.calls} == {"whisp-mute"}


def test_the_service_coalesces_requests() -> None:
    muter = FakeMuter()
    muter.leftovers_gate.clear()  # keep the worker busy while requests pile up
    service = MuteService(muter)
    try:
        for _ in range(3):
            service.mute()
            service.restore()
        service.mute()
        service.mute()
        muter.leftovers_gate.set()
        muter.wait_for(lambda names: "mute_others" in names)
        service.restore()
        service.restore()
        muter.wait_for(lambda names: "restore" in names)
    finally:
        service.shutdown(5)
    assert muter.names() == ["restore_leftovers", "mute_others", "restore", "restore"]


def test_a_mute_cancelled_before_it_ran_costs_nothing() -> None:
    muter = FakeMuter()
    muter.leftovers_gate.clear()
    service = MuteService(muter)
    service.mute()
    service.restore()
    muter.leftovers_gate.set()
    service.shutdown(5)
    assert muter.names() == ["restore_leftovers", "restore"]


def test_shutdown_restores_and_later_requests_are_ignored() -> None:
    muter = FakeMuter()
    service = MuteService(muter)
    service.mute()
    muter.wait_for(lambda names: "mute_others" in names)
    service.shutdown(5)
    assert muter.names() == ["restore_leftovers", "mute_others", "restore"]
    service.mute()
    service.restore()
    service.set_extra_exclusions(["obs64"])
    service.shutdown(5)
    assert muter.names() == ["restore_leftovers", "mute_others", "restore"]


def test_shutdown_waits_no_longer_than_its_timeout(caplog: pytest.LogCaptureFixture) -> None:
    muter = FakeMuter()
    muter.restore_gate.clear()  # Windows audio hangs
    service = MuteService(muter)
    muter.wait_for(lambda names: "restore_leftovers" in names)
    started = time.monotonic()
    service.shutdown(0.2)
    assert time.monotonic() - started < 2.0
    assert "still running" in caplog.text
    muter.restore_gate.set()
    muter.wait_for(lambda names: "restore" in names)


def test_the_service_survives_errors(caplog: pytest.LogCaptureFixture) -> None:
    muter = FakeMuter()
    muter.failing = {"restore_leftovers", "mute_others"}
    service = MuteService(muter)
    try:
        service.mute()
        muter.wait_for(lambda names: "mute_others" in names)
        service.restore()
        muter.wait_for(lambda names: "restore" in names)
    finally:
        service.shutdown(5)
    assert muter.names() == ["restore_leftovers", "mute_others", "restore", "restore"]
    assert "Couldn't mute other apps" in caplog.text
    assert "Couldn't unmute apps left muted by an earlier run" in caplog.text


def test_pending_sessions_are_checked_only_between_dictations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_mute, "PENDING_CHECK_FAST_S", 0.01)
    muter = FakeMuter()
    muter.has_pending = True
    service = MuteService(muter)
    try:
        muter.wait_for(lambda names: names.count("check_pending") >= 3)
        service.mute()
        muter.wait_for(lambda names: "mute_others" in names)
        time.sleep(0.2)  # long enough for many checks, if any were (wrongly) made mid-dictation
        service.restore()
        muter.wait_for(lambda names: "restore" in names)
        muter.wait_for(lambda names: names[-1] == "check_pending")
        muter.has_pending = False
    finally:
        service.shutdown(5)
    names = muter.names()
    start = names.index("mute_others")
    assert "check_pending" not in names[start : names.index("restore", start)]


def test_quick_checks_slow_down_and_restart_after_each_dictation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_mute, "PENDING_CHECK_FAST_S", 0.01)
    monkeypatch.setattr(audio_mute, "PENDING_CHECK_SLOW_S", 60.0)
    monkeypatch.setattr(audio_mute, "PENDING_FAST_PERIOD_S", 0.1)
    muter = FakeMuter()
    muter.has_pending = True
    service = MuteService(muter)
    try:
        muter.wait_for(lambda names: names.count("check_pending") >= 2)
        # Past the quick period the next check is a minute away, so the count stops growing.
        slowed, deadline = -1, time.monotonic() + 5.0
        while (count := muter.names().count("check_pending")) != slowed:
            assert time.monotonic() < deadline, "the checks never slowed down"
            slowed = count
            time.sleep(0.3)
        service.mute()
        muter.wait_for(lambda names: "mute_others" in names)
        service.restore()
        muter.wait_for(lambda names: names.count("check_pending") >= slowed + 2)
    finally:
        service.shutdown(5)


def test_no_checks_when_nothing_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_mute, "PENDING_CHECK_FAST_S", 0.01)
    muter = FakeMuter()
    service = MuteService(muter)
    muter.wait_for(lambda names: "restore_leftovers" in names)
    time.sleep(0.2)
    service.shutdown(5)
    assert "check_pending" not in muter.names()


def test_exclusions_reach_the_muter_on_its_thread() -> None:
    muter = FakeMuter()
    service = MuteService(muter, extra_exclusions=["obs64"])
    try:
        muter.wait_for(lambda names: "set_extra_exclusions" in names)
        assert muter.exclusions == ("obs64",)
        service.set_extra_exclusions(["Audacity.exe", "reaper"])
        muter.wait_for(lambda names: names.count("set_extra_exclusions") == 2)
        assert muter.exclusions == ("Audacity.exe", "reaper")
    finally:
        service.shutdown(5)
    assert {thread for _, thread in muter.calls} == {"whisp-mute"}


def test_the_real_muter_starts_and_stops_cleanly() -> None:
    service = MuteService()
    service.shutdown(5)
    assert not service._thread.is_alive()
    assert not (app_paths().data_dir / STATE_FILE_NAME).exists()


# -- real Windows audio sessions (desktop) -----------------------------------------------------


CHILD_SCRIPT = """
import os, sys, threading
try:
    import sounddevice as sd
    stream = sd.OutputStream(channels=1, dtype="float32", callback=lambda outdata, *_: outdata.fill(0))
    stream.start()
except Exception as exc:
    print("error:", type(exc).__name__, str(exc).replace("\\n", " "), flush=True)
    raise SystemExit(1)
print(os.getpid(), flush=True)
stop = threading.Event()
threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
stop.wait(float(sys.argv[1]))
stream.close()
"""
"""Plays digital silence until told to quit (a line on stdin, or stdin closing) or its time is up. The first line
it prints is its pid, or the error that stopped it from playing."""


def _readline(stream: IO[str], timeout: float) -> str:
    box: list[str] = []
    reader = threading.Thread(target=lambda: box.append(stream.readline()), daemon=True)
    reader.start()
    reader.join(timeout)
    return box[0] if box else ""


class SilentChild:
    """A separate Python process playing digital silence: an audio session that is ours to mute.

    (A process can't be muted by its own exclusion rules, which is why it has to be another process.)
    """

    def __init__(self, lifetime_s: float = 60.0) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", CHILD_SCRIPT, str(lifetime_s)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.pid = 0

    def wait_ready(self) -> int:
        assert self.proc.stdout is not None
        line = _readline(self.proc.stdout, 20.0).strip()
        if not line.isdigit():
            self.stop()
            pytest.skip(f"A child process couldn't play audio ({line or 'no answer within 20 s'})")
        self.pid = int(line)
        return self.pid

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.running and self.proc.stdin is not None:
            with contextlib.suppress(OSError):
                self.proc.stdin.write("quit\n")
                self.proc.stdin.close()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(5)


class OnlyThesePids:
    """The real sessions, narrowed to the test's own child processes, so nothing else on the machine is touched."""

    def __init__(self, real: CoreAudioSessions, pids: set[int]) -> None:
        self.real = real
        self.pids = pids

    @contextlib.contextmanager
    def snapshot(self, *, details: bool) -> Iterator[SessionSnapshot]:
        with self.real.snapshot(details=details) as snap:
            ours = [s for s in snap.sessions if s.pid in self.pids]
            yield SessionSnapshot(sessions=ours, capturing_apps=snap.capturing_apps)


def _sessions_of(real: CoreAudioSessions, pid: int) -> list[tuple[bool, bool]]:
    """(active, muted) of each session the process has."""
    with real.snapshot(details=False) as snap:
        return [(s.active, s.muted) for s in snap.sessions if s.pid == pid]


def _is_muted(real: CoreAudioSessions, pid: int) -> bool:
    sessions = _sessions_of(real, pid)
    assert sessions, f"process {pid} has no audio session"
    return all(muted for _, muted in sessions)


def _set_mute_as_user(real: CoreAudioSessions, pid: int, muted: bool) -> None:
    """Change a session's mute the way the volume mixer does: without Whisp's tag."""
    with real.snapshot(details=False) as snap:
        for session in snap.sessions:
            if session.pid == pid:
                assert audio_mute._SetMute(session.volume, muted, None) >= 0


def _wait_until(condition: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f"Timed out waiting until {what}")
        time.sleep(0.05)


def _run_on_com_thread(scenario: Callable[[], None], timeout: float = 120.0) -> None:
    """Run on a fresh thread (as Whisp does), leaving the test thread's COM apartment untouched for Qt."""
    box: dict[str, BaseException] = {}

    def target() -> None:
        try:
            scenario()
        except BaseException as exc:  # re-raised on the test thread
            box["error"] = exc
        finally:
            audio_mute._end_com()

    thread = threading.Thread(target=target, name="test-audio-mute", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail("The audio session scenario hung")
    if "error" in box:
        raise box["error"]


@pytest.mark.desktop
def test_mutes_and_restores_real_audio_sessions(tmp_path: Path) -> None:
    _run_on_com_thread(lambda: _real_session_scenario(tmp_path))


def _real_session_scenario(tmp_path: Path) -> None:
    real = CoreAudioSessions()
    with real.snapshot(details=False) as snap:
        baseline = {s.instance_id: s.muted for s in snap.sessions}
    children: list[SilentChild] = []
    pids: set[int] = set()
    state_path = tmp_path / STATE_FILE_NAME
    # Windows saves python.exe's mute when one of the children exits muted; the cleanup makes sure it's off again.
    saved_mute_may_stick = False

    def start_children(count: int) -> list[SilentChild]:
        started = [SilentChild() for _ in range(count)]
        children.extend(started)
        for child in started:
            pids.add(child.wait_ready())
        _wait_until(
            lambda: all(any(active for active, _ in _sessions_of(real, c.pid)) for c in started),
            10.0,
            "the children's audio sessions were playing",
        )
        return started

    try:
        music, user_muted = start_children(2)
        _set_mute_as_user(real, user_muted.pid, True)
        source = OnlyThesePids(real, pids)
        muter = AudioMuter(state_file=state_path, source=source)

        # A dictation: the playing child is muted, the one the user muted is left alone.
        assert muter.mute_others() == 1
        assert _is_muted(real, music.pid) and _is_muted(real, user_muted.pid)
        entries = json.loads(state_path.read_text(encoding="utf-8"))["sessions"]
        assert [(e["pid"], e["process"]) for e in entries] == [(music.pid, "python.exe")]
        assert muter.restore() == 1
        assert not _is_muted(real, music.pid)
        assert _is_muted(real, user_muted.pid)
        assert not state_path.exists()

        # The user unmutes and mutes again mid-dictation: Windows tells us, and the restore leaves it muted.
        assert muter.mute_others() == 1
        _set_mute_as_user(real, music.pid, False)
        _set_mute_as_user(real, music.pid, True)
        _wait_until(
            lambda: any(h.watch is not None and h.watch.changed_by_others for h in muter._held.values()),
            5.0,
            "Windows reported the user's change",
        )
        assert muter.restore() == 0
        assert _is_muted(real, music.pid)
        _set_mute_as_user(real, music.pid, False)

        # Whisp crashes mid-dictation: the next start unmutes from the saved list.
        crashed = AudioMuter(state_file=state_path, source=source)
        assert crashed.mute_others() == 1
        assert _is_muted(real, music.pid)
        assert AudioMuter(state_file=state_path, source=source).restore_leftovers() == 1
        assert not _is_muted(real, music.pid)
        assert not state_path.exists()
        assert crashed.restore() == 0  # only releases what it was watching

        # The app quits while muted, so Windows starts its next instance muted; that one is unmuted when it appears.
        assert muter.mute_others() == 1
        saved_mute_may_stick = True
        music.stop()
        assert muter.restore() == 0
        assert muter.has_pending
        assert state_path.exists()
        (relaunched,) = start_children(1)
        started_muted = _is_muted(real, relaunched.pid)
        assert muter.check_pending() == (1 if started_muted else 0)
        assert not _is_muted(real, relaunched.pid)
        saved_mute_may_stick = False
        assert _is_muted(real, user_muted.pid)  # it already existed, so its mute stays the user's
        assert not muter.has_pending
        assert not state_path.exists()

        # The service does the same on its own thread, and shutting down unmutes.
        service = MuteService(AudioMuter(state_file=state_path, source=source))
        try:
            service.mute()
            _wait_until(lambda: _is_muted(real, relaunched.pid), 5.0, "the service muted the child")
        finally:
            service.shutdown(5)
        assert not _is_muted(real, relaunched.pid)
        assert not state_path.exists()

        # Every watch was closed, so COM let go of every event sink.
        _wait_until(lambda: not audio_mute._sinks, 5.0, "COM released the event sinks")
    finally:
        # Each step runs even if an earlier one fails: nothing may be left muted on the owner's machine.
        for child in children:
            if child.running and child.pid:
                try:
                    _set_mute_as_user(real, child.pid, False)  # before it exits, so Windows saves it unmuted
                except BaseException:
                    saved_mute_may_stick = True
        for child in children:
            child.stop()
        if saved_mute_may_stick:
            _clear_saved_python_mute(real)
        _put_back(real, baseline)


def _clear_saved_python_mute(real: CoreAudioSessions) -> None:
    """Start one more child; if Windows starts it muted, unmute it, so python.exe is saved unmuted again."""
    probe = SilentChild()
    try:
        pid = probe.wait_ready()
        _wait_until(lambda: bool(_sessions_of(real, pid)), 10.0, "the probe had an audio session")
        _set_mute_as_user(real, pid, False)
    finally:
        probe.stop()


def _put_back(real: CoreAudioSessions, baseline: dict[str, bool]) -> None:
    """Return every session that existed before the test to the mute it had then."""
    with real.snapshot(details=False) as snap:
        for session in snap.sessions:
            if session.instance_id in baseline and session.muted != baseline[session.instance_id]:
                session.set_mute(baseline[session.instance_id])
