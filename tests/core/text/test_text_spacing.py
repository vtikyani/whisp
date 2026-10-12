"""Smart spacing between consecutive dictations."""

from __future__ import annotations

import threading

import pytest

from whisp.core.text.spacing import AUTO_WINDOW_S, WINDOWS_REMEMBERED, SpacingMemory
from whisp.core.types import KeyActivity, KeyKind, TargetWindow


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def window(hwnd: int = 10, pid: int = 20) -> TargetWindow:
    return TargetWindow(hwnd=hwnd, pid=pid, process_name="notepad.exe", app_name="Notepad", title="")


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def memory(clock: Clock) -> SpacingMemory:
    return SpacingMemory(clock=clock)


def test_auto_adds_a_space_after_a_recent_dictation_in_the_same_window(memory: SpacingMemory, clock: Clock) -> None:
    memory.remember(window(), "Hello there.")
    clock.now += 5
    assert memory.prefix_for(window(), "How are you?", "auto") == " "


def test_auto_without_history(memory: SpacingMemory) -> None:
    assert memory.prefix_for(window(), "Hello", "auto") == ""


def test_auto_other_window_or_process(memory: SpacingMemory) -> None:
    memory.remember(window(hwnd=10), "Hello.")
    assert memory.prefix_for(window(hwnd=11), "Next", "auto") == ""
    assert memory.prefix_for(window(hwnd=10, pid=99), "Next", "auto") == ""  # window handle reused by another app


def test_auto_remembers_each_window(memory: SpacingMemory) -> None:
    # An email, then a chat message, then back to the email: its cursor is still after "Hello."
    memory.remember(window(hwnd=10), "Hello.")
    memory.remember(window(hwnd=11), "Elsewhere.")
    assert memory.prefix_for(window(hwnd=10), "Back again", "auto") == " "
    assert memory.prefix_for(window(hwnd=11), "And here", "auto") == " "


def test_the_window_dictated_into_least_recently_is_forgotten_first(memory: SpacingMemory) -> None:
    for hwnd in range(1, WINDOWS_REMEMBERED + 2):
        memory.remember(window(hwnd=hwnd), "Hello.")
    assert memory.prefix_for(window(hwnd=1), "Next", "auto") == ""
    assert memory.prefix_for(window(hwnd=2), "Next", "auto") == " "
    assert memory.prefix_for(window(hwnd=WINDOWS_REMEMBERED + 1), "Next", "auto") == " "


def test_auto_expires_after_ten_minutes(memory: SpacingMemory, clock: Clock) -> None:
    memory.remember(window(), "Hello.")
    clock.now += AUTO_WINDOW_S - 1
    assert memory.prefix_for(window(), "Still here", "auto") == " "
    clock.now += 2
    assert memory.prefix_for(window(), "Too late", "auto") == ""


@pytest.mark.parametrize("previous", ["Hello\n", "Hello ", "List:\n\n", "Call (", "See “"])
def test_auto_no_space_after_open_endings(memory: SpacingMemory, previous: str) -> None:
    memory.remember(window(), previous)
    assert memory.prefix_for(window(), "next", "auto") == ""


@pytest.mark.parametrize("text", [", and more", ". Done", "! wow", "? really", "\nNew line", ")", "'s car", "%"])
def test_no_space_before_punctuation_or_a_line_break(memory: SpacingMemory, text: str) -> None:
    memory.remember(window(), "Hello")
    assert memory.prefix_for(window(), text, "auto") == ""
    assert memory.prefix_for(window(), text, "always") == ""


def test_no_space_next_to_chinese_or_japanese(memory: SpacingMemory) -> None:
    memory.remember(window(), "你好。")
    assert memory.prefix_for(window(), "我是小明。", "auto") == ""
    assert memory.prefix_for(window(), "Hello", "auto") == ""
    memory.remember(window(), "Hello.")
    assert memory.prefix_for(window(), "今日は", "auto") == ""
    assert memory.prefix_for(window(), "今日は", "always") == ""


def test_always_and_never(memory: SpacingMemory) -> None:
    assert memory.prefix_for(None, "Hello", "always") == " "
    assert memory.prefix_for(window(), "Hello", "always") == " "
    memory.remember(window(), "Hello.")
    assert memory.prefix_for(window(), "Again", "never") == ""


def test_unknown_target_or_mode(memory: SpacingMemory) -> None:
    memory.remember(window(), "Hello.")
    assert memory.prefix_for(None, "Next", "auto") == ""
    assert memory.prefix_for(window(), "Next", "sometimes") == ""
    assert memory.prefix_for(window(), "", "always") == ""


def test_remembering_without_a_target_forgets(memory: SpacingMemory) -> None:
    memory.remember(window(hwnd=10), "Hello.")
    memory.remember(window(hwnd=11), "Hi.")
    memory.remember(None, "Somewhere unknown.")  # maybe one of those two windows
    assert memory.prefix_for(window(hwnd=10), "Next", "auto") == ""
    assert memory.prefix_for(window(hwnd=11), "Next", "auto") == ""


def test_forget(memory: SpacingMemory) -> None:
    memory.remember(window(hwnd=10), "Hello.")
    memory.remember(window(hwnd=11), "Hi.")
    memory.forget()
    assert memory.prefix_for(window(hwnd=10), "Next", "auto") == ""
    assert memory.prefix_for(window(hwnd=11), "Next", "auto") == ""


# -- with keyboard activity (what HotkeyService.typing_activity reports) --------------------------


class Keys:
    """Stands in for HotkeyService.typing_activity: the kind of the user's last key in each window, nothing more."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.serial = 0
        self.last: dict[int, KeyActivity] = {}
        self.baseline = KeyActivity(serial=0, kind=KeyKind.MOVE, hwnd=0, at=float("-inf"))

    def type(self, kind: KeyKind, hwnd: int = 10) -> None:
        self.serial += 1
        self.last[hwnd] = KeyActivity(serial=self.serial, kind=kind, hwnd=hwnd, at=self.clock.now)

    def restart(self) -> None:
        """What the hook reports after it was off or Windows dropped it: keys may have gone unseen."""
        self.serial += 1
        self.last.clear()
        self.baseline = KeyActivity(serial=self.serial, kind=KeyKind.MOVE, hwnd=0, at=float("-inf"))

    def __call__(self, hwnd: int) -> KeyActivity | None:
        return self.last.get(hwnd, self.baseline)


@pytest.fixture
def keys(clock: Clock) -> Keys:
    return Keys(clock)


@pytest.fixture
def aware(clock: Clock, keys: Keys) -> SpacingMemory:
    return SpacingMemory(clock=clock, activity=keys)


def test_back_to_back_dictations_get_a_space(aware: SpacingMemory, clock: Clock) -> None:
    aware.remember(window(), "Hello there.")
    clock.now += 5
    assert aware.prefix_for(window(), "How are you?", "auto") == " "


def test_pressing_enter_between_dictations_means_no_space(aware: SpacingMemory, keys: Keys, clock: Clock) -> None:
    # The chat case: dictate, press Enter to send, dictate the next message.
    aware.remember(window(), "On my way.")
    keys.type(KeyKind.NEWLINE)
    clock.now += 3
    assert aware.prefix_for(window(), "See you soon.", "auto") == ""


@pytest.mark.parametrize("kind", [KeyKind.SPACE, KeyKind.EDIT, KeyKind.MOVE, KeyKind.NEWLINE])
def test_space_editing_or_moving_the_cursor_means_no_space(aware: SpacingMemory, keys: Keys, kind: KeyKind) -> None:
    aware.remember(window(), "First part.")
    keys.type(kind)
    assert aware.prefix_for(window(), "Second part.", "auto") == ""


def test_a_word_typed_just_before_dictating_gets_a_space(aware: SpacingMemory, keys: Keys) -> None:
    keys.type(KeyKind.TEXT)
    assert aware.prefix_for(window(), "and then the rest", "auto") == " "


def test_typing_after_a_dictation_then_dictating_again(aware: SpacingMemory, keys: Keys) -> None:
    aware.remember(window(), "Dictated.")
    keys.type(KeyKind.SPACE)
    keys.type(KeyKind.TEXT)  # the user typed " Typed" by hand
    assert aware.prefix_for(window(), "Dictated again.", "auto") == " "


def test_typing_in_another_window_doesnt_count(aware: SpacingMemory, keys: Keys) -> None:
    keys.type(KeyKind.TEXT, hwnd=99)
    assert aware.prefix_for(window(hwnd=10), "Hello", "auto") == ""


def test_working_in_another_app_between_dictations_keeps_the_space(aware: SpacingMemory, keys: Keys) -> None:
    # Dictate into an email, answer a chat by hand, come back: the email's cursor hasn't moved.
    aware.remember(window(hwnd=10), "Hello.")
    keys.type(KeyKind.TEXT, hwnd=99)
    keys.type(KeyKind.NEWLINE, hwnd=99)
    assert aware.prefix_for(window(hwnd=10), "How are you?", "auto") == " "


def test_keys_typed_in_the_window_after_coming_back_still_count(aware: SpacingMemory, keys: Keys) -> None:
    aware.remember(window(hwnd=10), "Hello.")
    keys.type(KeyKind.TEXT, hwnd=99)
    keys.type(KeyKind.NEWLINE, hwnd=10)
    assert aware.prefix_for(window(hwnd=10), "Next paragraph", "auto") == ""


def test_keys_the_hook_may_have_missed_mean_no_space(aware: SpacingMemory, keys: Keys) -> None:
    aware.remember(window(), "Hello.")
    keys.restart()  # the user may have typed anything meanwhile
    assert aware.prefix_for(window(), "Next", "auto") == ""
    aware.remember(window(), "Next")
    assert aware.prefix_for(window(), "and more", "auto") == " "  # watched again since that dictation


def test_old_typing_doesnt_count(aware: SpacingMemory, keys: Keys, clock: Clock) -> None:
    keys.type(KeyKind.TEXT)
    clock.now += AUTO_WINDOW_S + 1
    assert aware.prefix_for(window(), "Hello", "auto") == ""


def test_nothing_typed_and_nothing_dictated_yet(aware: SpacingMemory) -> None:
    assert aware.prefix_for(window(), "Hello", "auto") == ""


def test_open_endings_still_win_when_nothing_was_typed(aware: SpacingMemory) -> None:
    aware.remember(window(), "Shopping list:\n")
    assert aware.prefix_for(window(), "milk", "auto") == ""


def test_unknown_keyboard_activity_falls_back_to_timing(clock: Clock) -> None:
    memory = SpacingMemory(clock=clock, activity=lambda hwnd: None)  # hook not installed
    memory.remember(window(), "Hello.")
    assert memory.prefix_for(window(), "Next", "auto") == " "


def test_a_failing_activity_source_falls_back_to_timing(clock: Clock) -> None:
    def broken(hwnd: int) -> KeyActivity | None:
        raise RuntimeError("hook gone")

    memory = SpacingMemory(clock=clock, activity=broken)
    memory.remember(window(), "Hello.")
    assert memory.prefix_for(window(), "Next", "auto") == " "


def test_thread_safe() -> None:
    memory = SpacingMemory()
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(2000):
                target = window(hwnd=n)
                memory.remember(target, f"text {i}.")
                memory.prefix_for(target, "next", "auto")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
