import logging
import time
from pynput.keyboard import Controller

log = logging.getLogger("whisp")


class Typer:
    """Simulates keystrokes to type text into the focused window."""

    def __init__(self, delay_ms=10):
        self.delay_s = delay_ms / 1000.0
        self._controller = Controller()

    def type_text(self, text: str):
        """Type text character by character with delay."""
        for char in text:
            self._controller.type(char)
            if self.delay_s > 0:
                time.sleep(self.delay_s)
