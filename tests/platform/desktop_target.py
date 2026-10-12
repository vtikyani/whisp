"""A test window in a separate process, for hotkey tests that need the focus outside the test process.

Run as ``python desktop_target.py [hwnd-to-focus-on-quit]``. One line per message on stdin/stdout:

    -> ready <hwnd>            the window has the focus (or "refused" if Windows wouldn't give it)
    <- keys <vk>:<d|u> ...     inject these keys — only while this window has the focus
    -> ok | lost-focus | rejected
    <- quit
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from desktop_support import press, release, set_foreground
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QLineEdit

from whisp.platform.win32 import keys
from whisp.platform.win32.foreground import foreground_window


def reply(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


class Commands(QObject):
    received = Signal(str)


def main() -> int:
    app = QApplication(sys.argv)
    window = QLineEdit()
    window.setWindowTitle("Whisp test target (separate process)")
    window.resize(360, 60)
    window.show()
    hwnd = int(window.winId())
    return_focus_to = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    commands = Commands()

    def handle(line: str) -> None:
        parts = line.split()
        if not parts or parts[0] == "quit":
            if return_focus_to and foreground_window() == hwnd:
                set_foreground(return_focus_to)
            app.quit()
            return
        if parts[0] == "keys":
            if foreground_window() != hwnd:
                reply("lost-focus")
                return
            events = []
            for item in parts[1:]:
                vk_text, action = item.split(":")
                vk = int(vk_text, 0)
                events.append(release(vk) if action == "u" else press(vk))
            reply("ok" if keys.send_input(events) == len(events) else "rejected")

    commands.received.connect(handle)

    def read_stdin() -> None:
        for line in sys.stdin:
            commands.received.emit(line.strip())
        commands.received.emit("quit")

    def focus() -> None:
        window.raise_()
        window.activateWindow()
        set_foreground(hwnd)
        QTimer.singleShot(300, lambda: reply(f"ready {hwnd}" if foreground_window() == hwnd else "refused"))
        threading.Thread(target=read_stdin, daemon=True).start()

    QTimer.singleShot(200, focus)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
