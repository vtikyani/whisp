"""Frame-rate independent animation primitives for the overlay.

Everything here advances by an explicit time step in seconds, so the same code
runs from the window's frame timer, from tests and from the render script, and
a dropped frame never changes how long an animation takes.
"""

from __future__ import annotations

import math
from collections.abc import Iterator


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_cubic(t: float) -> float:
    t = clamp(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp(t)
    return 4.0 * t * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


class Spring:
    """A critically damped spring: the fastest approach to a target that never overshoots from rest.

    ``omega`` is the natural frequency in rad/s; the value is within ~1% of its
    target after about ``6.6 / omega`` seconds. Retargeting keeps the current
    value and velocity, so a morph can change direction mid-flight without a jump.
    """

    __slots__ = ("omega", "precision", "target", "value", "velocity")

    def __init__(self, value: float, omega: float = 20.0, precision: float = 0.01) -> None:
        self.value = float(value)
        self.target = float(value)
        self.velocity = 0.0
        self.omega = omega
        self.precision = precision

    def snap(self, value: float | None = None) -> None:
        """Jump straight to ``value`` (default: the current target) and stop."""
        if value is not None:
            self.target = float(value)
        self.value = self.target
        self.velocity = 0.0

    def step(self, dt: float) -> None:
        if dt <= 0.0 or self.settled:
            return
        # Exact solution of x'' = -w^2 x - 2w x' over dt: stable for any step size.
        displacement = self.value - self.target
        temp = (self.velocity + self.omega * displacement) * dt
        decay = math.exp(-self.omega * dt)
        self.value = self.target + (displacement + temp) * decay
        self.velocity = (self.velocity - self.omega * temp) * decay
        if abs(self.value - self.target) < self.precision and abs(self.velocity) < self.precision * 4.0:
            self.snap()

    @property
    def settled(self) -> bool:
        return self.value == self.target and self.velocity == 0.0


class Fade:
    """Opacity that moves at a constant speed toward shown/hidden and is read through an ease curve.

    Constant speed makes reversals seamless: fading out halfway and back in
    takes exactly as long as the distance travelled.
    """

    __slots__ = ("_wait", "delay", "fade_in", "fade_out", "progress", "target")

    def __init__(
        self, visible: bool = False, fade_in: float = 0.18, fade_out: float = 0.12, delay: float = 0.0
    ) -> None:
        self.progress = 1.0 if visible else 0.0
        self.target = self.progress
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.delay = delay
        """Seconds to wait before fading in from fully hidden (lets other elements make room first)."""
        self._wait = 0.0

    def show(self, visible: bool = True, *, instantly: bool = False) -> None:
        target = 1.0 if visible else 0.0
        if target > self.target and self.progress == 0.0:
            self._wait = self.delay
        elif not visible:
            self._wait = 0.0
        self.target = target
        if instantly:
            self.progress = self.target
            self._wait = 0.0

    def step(self, dt: float) -> None:
        if self.progress == self.target or dt <= 0.0:
            return
        if self._wait > 0.0 and self.target > self.progress:
            used = min(self._wait, dt)
            self._wait -= used
            dt -= used
            if dt <= 0.0:
                return
        if self.target > self.progress:
            self.progress = min(self.target, self.progress + dt / max(self.fade_in, 1e-6))
        else:
            self.progress = max(self.target, self.progress - dt / max(self.fade_out, 1e-6))

    @property
    def value(self) -> float:
        return smoothstep(self.progress)

    @property
    def visible(self) -> bool:
        return self.target > 0.0

    @property
    def settled(self) -> bool:
        return self.progress == self.target


class Slot[T]:
    """Holds one value at a time and cross-fades whenever the value changes.

    Old values keep painting while they fade out, so content never pops.
    Iterating yields ``(value, opacity)`` pairs, oldest first.
    """

    __slots__ = ("_delay", "_fade_in", "_fade_out", "_layers")

    def __init__(self, fade_in: float = 0.2, fade_out: float = 0.12, delay: float = 0.0) -> None:
        self._fade_in = fade_in
        self._fade_out = fade_out
        self._delay = delay
        self._layers: list[tuple[T, Fade]] = []

    @property
    def current(self) -> T | None:
        if self._layers and self._layers[-1][1].visible:
            return self._layers[-1][0]
        return None

    def set(self, value: T | None, *, instantly: bool = False) -> None:
        if value is not None and self.current is not None and self.current == value:
            return
        for _, fade in self._layers:
            fade.show(False, instantly=instantly)
        if value is not None:
            fade = Fade(False, self._fade_in, self._fade_out, delay=self._delay)
            fade.show(True, instantly=instantly)
            self._layers.append((value, fade))
        self._prune()

    def step(self, dt: float) -> None:
        for _, fade in self._layers:
            fade.step(dt)
        self._prune()

    def _prune(self) -> None:
        self._layers = [(v, f) for v, f in self._layers if f.visible or f.progress > 0.0]

    @property
    def settled(self) -> bool:
        return all(f.settled for _, f in self._layers)

    @property
    def opacity(self) -> float:
        """Opacity of the current value (0 when there is none)."""
        if self._layers and self._layers[-1][1].visible:
            return self._layers[-1][1].value
        return 0.0

    def __iter__(self) -> Iterator[tuple[T, float]]:
        for value, fade in self._layers:
            opacity = fade.value
            if opacity > 0.0:
                yield value, opacity


class LevelFollower:
    """Envelope follower for the microphone level: rises fast (attack) and falls slowly (release).

    ``attack`` and ``release`` are time constants in seconds.
    """

    __slots__ = ("attack", "release", "target", "value")

    def __init__(self, attack: float = 0.06, release: float = 0.25) -> None:
        self.attack = attack
        self.release = release
        self.value = 0.0
        self.target = 0.0

    def set(self, level: float) -> None:
        # NaN compares False everywhere; treat it as silence instead of poisoning the animation.
        self.target = clamp(level) if level == level else 0.0

    def reset(self) -> None:
        self.value = self.target = 0.0

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            return
        tau = self.attack if self.target > self.value else self.release
        self.value += (self.target - self.value) * (1.0 - math.exp(-dt / tau))
        if abs(self.value - self.target) < 1e-4:
            self.value = self.target
