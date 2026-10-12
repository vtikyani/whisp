"""ONNX Runtime reaches Whisp only through ``whisp.core.onnx_runtime``, which switches its telemetry off."""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path

import pytest

import whisp
from whisp.core import onnx_runtime


def test_importing_it_switches_telemetry_off(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(onnx_runtime.ort, "disable_telemetry_events", lambda: calls.append("off"))
    importlib.reload(onnx_runtime)
    assert calls == ["off"]
    assert onnx_runtime.ort.__name__ == "onnxruntime"


def test_a_runtime_without_the_switch_is_logged_not_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delattr(onnx_runtime.ort, "disable_telemetry_events")
    with caplog.at_level(logging.WARNING, logger="whisp.core.onnx_runtime"):
        importlib.reload(onnx_runtime)
    assert "telemetry" in caplog.text


def test_whisp_imports_onnx_runtime_only_through_it() -> None:
    root = Path(whisp.__file__).parent
    direct = re.compile(r"^\s*(?:import onnxruntime\b|from onnxruntime\b)", re.MULTILINE)
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.name != "onnx_runtime.py" and direct.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
