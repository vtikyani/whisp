"""ONNX Runtime, with its telemetry switched off. Every part of Whisp imports it from here.

Official Windows builds of ONNX Runtime report usage through a Windows
TraceLogging provider, and it is on unless the program turns it off (see ONNX
Runtime's Privacy.md). Whisp promises no telemetry, so importing this module
turns it off before any model session exists. ONNX Runtime 1.31 reads no
environment variable for this; ``disable_telemetry_events()`` is the switch.
"""

from __future__ import annotations

import logging

import onnxruntime as ort

log = logging.getLogger(__name__)

__all__ = ["ort"]

_disable = getattr(ort, "disable_telemetry_events", None)
if _disable is None:
    log.warning("This ONNX Runtime (%s) has no switch to turn its telemetry off", ort.__version__)
else:
    _disable()
