import logging
from pynput import keyboard as pynput_kb
from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger("whisp")


class HotkeyListener(QObject):
    """Global hotkey listener using pynput. Emits signal on hotkey press."""

    toggled = pyqtSignal()

    def __init__(self, hotkey_str="ctrl+shift+space"):
        super().__init__()
        self._hotkey_str = hotkey_str.lower().strip()
        self._listener = None

    def _on_hotkey(self):
        log.debug("Hotkey pressed")
        self.toggled.emit()

    @staticmethod
    def _to_pynput_hotkey(hotkey_str):
        """Convert 'ctrl+shift+space' → '<ctrl>+<shift>+<space>'"""
        parts = hotkey_str.split('+')
        return '+'.join(f'<{p}>' if len(p) > 1 else p for p in parts)

    def start(self):
        pynput_hotkey = self._to_pynput_hotkey(self._hotkey_str)
        self._listener = pynput_kb.GlobalHotKeys({pynput_hotkey: self._on_hotkey})
        self._listener.start()
        log.info("Hotkey listener started: %s", self._hotkey_str)

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            log.info("Hotkey listener stopped")
