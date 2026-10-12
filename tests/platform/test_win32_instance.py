"""Single-instance lock and activation messages, with two SingleInstance objects in one process."""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Iterator

import pytest

from whisp.platform.win32.instance import SingleInstance, default_instance_name


def send_from_another_thread(qtbot, sender: SingleInstance, message: str) -> bool:
    """``send`` blocks, like a second launch does; the first instance's event loop must keep running meanwhile."""
    result: list[bool] = []
    thread = threading.Thread(target=lambda: result.append(sender.send(message)))
    thread.start()
    qtbot.waitUntil(lambda: not thread.is_alive(), timeout=5000)
    return result == [True]


@pytest.fixture
def name() -> str:
    # A unique name per test, so a real running Whisp (or a parallel test run) can't interfere.
    return f"WhispTest-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def instances(qapp, name: str) -> Iterator[tuple[SingleInstance, SingleInstance]]:
    first, second = SingleInstance(name), SingleInstance(name)
    try:
        yield first, second
    finally:
        second.release()
        first.release()


def test_default_name_is_stable_and_private() -> None:
    value = default_instance_name()
    assert value == default_instance_name()
    assert re.fullmatch(r"Whisp-[0-9a-f]{16}", value)
    assert SingleInstance().name == value


def test_only_the_first_acquires(instances: tuple[SingleInstance, SingleInstance]) -> None:
    first, second = instances
    assert first.acquire() is True
    assert first.acquire() is True  # idempotent for the owner
    assert second.acquire() is False


def test_second_launch_sends_a_message_to_the_first(qtbot, instances: tuple[SingleInstance, SingleInstance]) -> None:
    first, second = instances
    assert first.acquire()
    assert not second.acquire()
    with qtbot.waitSignal(first.message, timeout=3000) as blocker:
        assert send_from_another_thread(qtbot, second, "show")
    assert blocker.args == ["show"]


def test_several_messages_and_line_breaks(qtbot, instances: tuple[SingleInstance, SingleInstance]) -> None:
    first, second = instances
    first.acquire()
    received: list[str] = []
    first.message.connect(received.append)
    assert send_from_another_thread(qtbot, second, "show")
    assert send_from_another_thread(qtbot, second, "open\nsettings")
    qtbot.waitUntil(lambda: len(received) == 2, timeout=3000)
    assert received == ["show", "open settings"]


def test_send_without_a_running_instance_returns_false(qapp, name: str) -> None:
    lonely = SingleInstance(name)
    assert lonely.send("show", timeout_ms=300) is False


def test_release_lets_the_next_launch_take_over(qtbot, instances: tuple[SingleInstance, SingleInstance]) -> None:
    first, second = instances
    assert first.acquire()
    first.release()
    first.release()  # idempotent
    assert second.acquire() is True
    with qtbot.waitSignal(second.message, timeout=3000) as blocker:
        assert send_from_another_thread(qtbot, first, "show")
    assert blocker.args == ["show"]
