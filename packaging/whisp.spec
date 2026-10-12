# PyInstaller recipe for Whisp: a one-folder, windowed Whisp.exe in dist\Whisp\.
#
# Build with scripts\build.ps1. It syncs a build environment (build\env) with uv.lock, then runs:
#
#     build\env\Scripts\python.exe -m PyInstaller --noconfirm --clean packaging\whisp.spec
#
# What goes in, and what deliberately doesn't:
#
# * Whisp's own code, its assets (logo, Inter font, Silero VAD model) and the
#   web UI, with every licence file that comes with them.
# * Qt (PySide6) with Qt WebEngine: its helper process, resources and locales.
#   Qt modules Whisp never imports are left out.
# * ONNX Runtime with all of its own DLLs, including the CUDA provider.
# * NOT the NVIDIA CUDA/cuDNN libraries. They take 1.6 GB on disk, so packaged
#   builds download them on demand as the "GPU runtime pack"
#   (src/whisp/core/asr/gpu.py). The llama.cpp server for AI polish is also
#   downloaded on demand (src/whisp/core/llm/runtime.py).
# * Licence texts next to Whisp.exe: Whisp's licence, THIRD_PARTY_NOTICES.md,
#   the GPL/LGPL texts that Qt's licence requires, Python's licence and
#   ONNX Runtime's notices. The licence files of the bundled Python packages
#   stay in their *.dist-info folders under _internal.
#
# Qt is linked dynamically and kept as separate DLLs under _internal\PySide6,
# so users can replace the Qt libraries with their own build (LGPL-3.0).

import re
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata, get_package_paths
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

ROOT = Path(SPECPATH).resolve().parent
PACKAGE = ROOT / "src" / "whisp"
PACKAGING = ROOT / "packaging"

APP_NAME = "Whisp"
COMPANY = "Whisp contributors"
DESCRIPTION = "Whisp voice dictation"


def read_version() -> str:
    """``__version__`` from src/whisp/__init__.py, read as text so the spec never imports the app."""
    text = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"(\d+)\.(\d+)\.(\d+)"\s*$', text, re.MULTILINE)
    if match is None:
        raise SystemExit('src/whisp/__init__.py must define __version__ = "MAJOR.MINOR.PATCH"')
    return ".".join(match.groups())


VERSION = read_version()
VERSION_TUPLE = (*(int(part) for part in VERSION.split(".")), 0)

# -- what to analyse --------------------------------------------------------------------------

datas = [
    (str(PACKAGE / "assets"), "whisp/assets"),
    (str(PACKAGE / "ui" / "main_window" / "web"), "whisp/ui/main_window/web"),
]
# onnx-asr reads its feature-extraction models and filter banks with importlib.resources.
datas += collect_data_files("onnx_asr")

# Package metadata (and with it each package's licence files). onnx-asr reads its own
# version at import time; Whisp's diagnostics and About page read the others.
RUNTIME_DISTRIBUTIONS = (
    "onnx-asr",
    "onnxruntime-gpu",
    "numpy",
    "sounddevice",
    "cffi",
    "pydantic",
    "pydantic-core",
    "annotated-types",
    "typing-inspection",
    "typing-extensions",
    "httpx",
    "httpcore",
    "h11",
    "anyio",
    "idna",
    "certifi",
    "PySide6-Essentials",
    "PySide6-Addons",  # Qt WebChannel and Qt Positioning, which Qt WebEngine needs
    "PySide6-WebEngine",
    "shiboken6",
)
for distribution in RUNTIME_DISTRIBUTIONS:
    datas += copy_metadata(distribution)

EXCLUDES = [
    # Development and test tools.
    "pytest",
    "_pytest",
    "pytestqt",
    "pluggy",
    "iniconfig",
    "pygments",
    "setuptools",
    "pkg_resources",
    "_distutils_hack",
    "pip",
    "PyInstaller",
    # Standard-library parts Whisp never uses (some are only imported by NumPy's test helpers).
    "tkinter",
    "_tkinter",
    "turtle",
    "turtledemo",
    "idlelib",
    "lib2to3",
    "pydoc",
    "pydoc_data",
    "pdb",
    "doctest",
    "unittest",
    "ensurepip",
    "venv",
    "numpy.testing",
    # Optional extras of libraries Whisp uses: NumPy's show_config() can print YAML, and httpx has a
    # command-line client (built on click) that it only loads when it's installed.
    "yaml",
    "httpx._main",
    "click",
    # NVIDIA's CUDA wheels: packaged builds use the downloaded GPU runtime pack instead.
    "nvidia",
    # onnx-asr can fetch models from Hugging Face itself; Whisp downloads and verifies its own.
    "huggingface_hub",
    # Python bindings of Qt modules that Qt WebEngine links to but Whisp never imports. Their
    # hooks would also collect every QML module (Quick Controls, 3D, PDF, ...). The Qt DLLs
    # themselves are still collected, because Qt6WebEngineCore.dll needs them.
    "PySide6.QtQuick",
    "PySide6.QtQml",
    "PySide6.QtPositioning",
    "PySide6.QtOpenGL",
    # Qt modules Whisp doesn't use at all.
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtAxContainer",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtConcurrent",
    "PySide6.QtDBus",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtHelp",
    "PySide6.QtHttpServer",
    "PySide6.QtLocation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetworkAuth",
    "PySide6.QtNfc",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialBus",
    "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSql",
    "PySide6.QtStateMachine",
    "PySide6.QtSvgWidgets",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtUiTools",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebSockets",
    "PySide6.QtXml",
]

a = Analysis(
    [str(PACKAGE / "__main__.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    excludes=EXCLUDES,
    noarchive=False,
)

# -- trim what the hooks collect but Whisp never loads ------------------------------------------

_NVIDIA_DLL = re.compile(
    r"^(cudart|cublas|cublaslt|cudnn|cufft|curand|cusolver|cusparse|cupti|nvrtc|nvjitlink|nvtoolsext|nccl)[^/\\]*\.dll$",
    re.IGNORECASE,
)

# Qt plugins Whisp loads, by plugin folder. Folders not listed here are kept whole.
QT_PLUGINS = {
    "platforms": {"qwindows.dll"},  # the Windows desktop
    "imageformats": {"qsvg.dll", "qico.dll"},  # SVG logo and tray icons; PNG and BMP are built into Qt
    "platforminputcontexts": set(),  # on-screen keyboard plugin; Windows IME support lives in qwindows.dll
    "tls": set(),  # Qt's network stack is only used for local named pipes; downloads use Python's ssl
    "networkinformation": set(),
    "generic": set(),  # TUIO network touch input
}
# DLLs that only the plugins dropped above load.
QT_PLUGIN_ONLY_DLLS = {
    "pyside6/qt6pdf.dll",  # imageformats/qpdf.dll
    "pyside6/qt6virtualkeyboard.dll",  # platforminputcontexts/qtvirtualkeyboardplugin.dll
    # tls/qopensslbackend.dll. PyInstaller's Qt hook finds these wherever OpenSSL sits on PATH;
    # Python's own libssl-3.dll and libcrypto-3.dll (for its ssl module) are different files and stay.
    "libssl-3-x64.dll",
    "libcrypto-3-x64.dll",
}


def _parts(dest: str) -> tuple[str, ...]:
    return tuple(part.lower() for part in re.split(r"[/\\]", dest))


def keep(entry: tuple[str, str, str]) -> bool:
    parts = _parts(entry[0])
    name = parts[-1]
    if parts[0] == "nvidia" or _NVIDIA_DLL.match(name):
        return False  # comes from the GPU runtime pack at run time
    if parts[:2] == ("_sounddevice_data", "portaudio-binaries"):
        # The hook takes every PortAudio build (32-bit, ARM, macOS, ASIO); Whisp needs the 64-bit Windows one.
        return name in ("libportaudio64bit.dll", "readme.md")
    if "/".join(parts) in QT_PLUGIN_ONLY_DLLS:
        return False
    if parts[0] != "pyside6":
        return True
    if len(parts) == 4 and parts[1] == "plugins" and parts[2] in QT_PLUGINS:
        return name in QT_PLUGINS[parts[2]]
    if parts[1] == "resources" and ".debug." in name:
        return False  # only used by debug builds of Qt WebEngine
    # Qt's own UI translations go: Whisp never installs a QTranslator. (Qt WebEngine's locales stay.)
    return not (parts[1] == "translations" and len(parts) == 3 and name.endswith(".qm"))


a.binaries = [entry for entry in a.binaries if keep(entry)]
a.datas = [entry for entry in a.datas if keep(entry)]

_leftover = sorted(entry[0] for entry in a.binaries if _NVIDIA_DLL.match(_parts(entry[0])[-1]))
if _leftover:
    raise SystemExit(f"NVIDIA runtime DLLs must not be bundled, but these were collected: {_leftover}")

# -- build ------------------------------------------------------------------------------------

pyz = PYZ(a.pure)

version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=VERSION_TUPLE, prodvers=VERSION_TUPLE),
    kids=[
        StringFileInfo(
            [
                StringTable(
                    "040904B0",  # US English, Unicode
                    [
                        StringStruct("CompanyName", COMPANY),
                        StringStruct("FileDescription", APP_NAME),
                        StringStruct("FileVersion", VERSION),
                        StringStruct("InternalName", APP_NAME),
                        StringStruct("LegalCopyright", f"Copyright (c) 2026 {COMPANY}. MIT License."),
                        StringStruct("OriginalFilename", f"{APP_NAME}.exe"),
                        StringStruct("ProductName", APP_NAME),
                        StringStruct("ProductVersion", VERSION),
                        StringStruct("Comments", DESCRIPTION),
                    ],
                )
            ]
        ),
        VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
    ],
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    icon=str(PACKAGING / "whisp.ico"),
    version=version_info,
    console=False,
    uac_admin=False,
    uac_uiaccess=False,
    upx=False,  # compressed Qt DLLs load slower and trip antivirus heuristics
    debug=False,
    strip=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

# -- licence texts next to Whisp.exe ------------------------------------------------------------


def _python_licence() -> Path:
    """CPython's licence, which also covers the libraries it bundles (OpenSSL, libffi, SQLite, ...)."""
    for base in (Path(sys.base_prefix), Path(sys.base_exec_prefix)):
        candidate = base / "LICENSE.txt"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Couldn't find Python's LICENSE.txt under {sys.base_prefix}; it must ship with Whisp")


_ort_dir = Path(get_package_paths("onnxruntime")[1])
_notices = {
    "LICENSE.txt": ROOT / "LICENSE",
    "THIRD_PARTY_NOTICES.md": ROOT / "THIRD_PARTY_NOTICES.md",
    "licenses/GPL-3.0.txt": PACKAGING / "licenses" / "GPL-3.0.txt",
    "licenses/LGPL-3.0.txt": PACKAGING / "licenses" / "LGPL-3.0.txt",
    "licenses/Python.txt": _python_licence(),
    "licenses/ONNX-Runtime.txt": _ort_dir / "LICENSE",
    "licenses/ONNX-Runtime-ThirdPartyNotices.txt": _ort_dir / "ThirdPartyNotices.txt",
}
_app_dir = Path(DISTPATH) / APP_NAME
for _target, _source in _notices.items():
    if not _source.is_file():
        raise SystemExit(f"Licence file missing: {_source}")
    (_app_dir / _target).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_source, _app_dir / _target)
