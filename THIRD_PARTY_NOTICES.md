# Third-party notices

Whisp is free software under the MIT License (see `LICENSE`). It is built on
the work of many other projects. Thank you to everyone behind them.

This file lists:

1. [What ships with Whisp](#what-ships-with-whisp): everything inside the
   installer and the app folder.
2. [What Whisp downloads later](#what-whisp-downloads-later), and only when
   you ask for it: speech models, AI models, the built-in AI server and the
   GPU runtime pack.

Versions are the ones in Whisp 1.0.0.

## Where the licence texts are

In the installed app folder (by default `%LOCALAPPDATA%\Programs\Whisp`):

| File | What it is |
|---|---|
| `LICENSE.txt` | Whisp's own licence (MIT) |
| `THIRD_PARTY_NOTICES.md` | This file |
| `licenses\LGPL-3.0.txt`, `licenses\GPL-3.0.txt` | The GNU licences Qt and PySide6 are used under |
| `licenses\Python.txt` | Python's licence, including the libraries in Python's Windows build |
| `licenses\ONNX-Runtime.txt`, `licenses\ONNX-Runtime-ThirdPartyNotices.txt` | ONNX Runtime's licence and the notices of the code it includes |
| `_internal\<package>.dist-info\licenses\` | The licence of each bundled Python package |
| `_internal\whisp\assets\` and `_internal\whisp\ui\main_window\web\vendor\` | The licences of the font, the voice detection model and the web UI libraries |

You can also read most of them in Whisp under **Settings → About**.

In the source code, the same files live in `packaging/licenses/`,
`src/whisp/assets/` and `src/whisp/ui/main_window/web/vendor/`.

## What ships with Whisp

### Qt and Qt for Python (PySide6)

| | |
|---|---|
| Components | Qt 6.12 libraries, including Qt WebEngine; PySide6-Essentials 6.12.0, PySide6-Addons 6.12.0 (only Qt WebChannel and Qt Positioning), PySide6-WebEngine 6.12.0.140, shiboken6 6.12.0 |
| Licence | Offered under LGPL-3.0-only, GPL-2.0-only or GPL-3.0-only. Whisp uses them under the **GNU Lesser General Public License v3.0** (`licenses\LGPL-3.0.txt`, which builds on `licenses\GPL-3.0.txt`). |
| Copyright | Copyright (C) The Qt Company Ltd. and other contributors |
| Source code | Qt: <https://download.qt.io/official_releases/qt/>. Qt for Python: <https://download.qt.io/official_releases/QtForPython/> |

Qt WebEngine contains Chromium, which includes many third-party components
under their own licences (mostly BSD- and MIT-style). Qt lists them at
<https://doc.qt.io/qt-6/qtwebengine-licensing.html>.

#### Replacing the Qt libraries

Whisp is a "one-folder" build. Qt is never linked into `Whisp.exe`: every Qt
library is a separate DLL in `_internal\PySide6\`, loaded when Whisp starts.
The LGPL gives you the right to swap them for your own build, and you can:

1. Get or build Qt 6.12 for 64-bit Windows with MSVC 2022. Whisp uses Qt Core,
   Gui, Widgets, Network, Svg, PrintSupport, Qml, Quick, QuickWidgets, OpenGL,
   Positioning, WebChannel, WebEngineCore and WebEngineWidgets.
2. Quit Whisp (right-click the tray icon, then **Quit Whisp**).
3. In the app folder, replace the matching `Qt6*.dll` files in
   `_internal\PySide6\`. If you rebuilt Qt WebEngine, also replace
   `QtWebEngineProcess.exe`, the `resources\` folder and
   `translations\qtwebengine_locales\` next to them.
4. Start Whisp again.

The Python bindings (the `.pyd` files in `_internal\PySide6\` and
`_internal\shiboken6\`) must match your Qt build. If they don't, build
PySide6 from its source against your Qt and replace those files as well. You
can also run Whisp from its source code with any compatible PySide6 (see
`README.md`).

### Python

| | |
|---|---|
| Component | CPython 3.12 (`python312.dll` and the parts of the standard library Whisp uses) |
| Licence | Python Software Foundation License Version 2 |
| Copyright | Copyright (c) 2001-2023 Python Software Foundation. All Rights Reserved. |
| Includes | The libraries in Python's Windows build that Whisp uses, such as OpenSSL, libffi, bzip2, SQLite and the Microsoft Visual C++ runtime. Their licences are in `licenses\Python.txt`. |

### Python packages

| Package | Version | Licence | Copyright |
|---|---|---|---|
| ONNX Runtime (onnxruntime-gpu) | 1.31.0 | MIT | Copyright (c) Microsoft Corporation |
| onnx-asr | 0.12.0 | MIT | Copyright (c) 2025 Ilya Stupakov |
| NumPy | 2.5.4 | BSD-3-Clause | Copyright (c) 2005-2025, NumPy Developers |
| python-sounddevice | 0.5.6 | MIT | Copyright (c) 2015-2025 Matthias Geier |
| cffi (its `_cffi_backend` module) | 2.1.1 | MIT-0 | The cffi authors |
| Pydantic | 2.14.1 | MIT | Copyright (c) 2017 to present Pydantic Services Inc. and individual contributors |
| pydantic-core | 2.50.1 | MIT | Copyright (c) 2022 Samuel Colvin |
| annotated-types | 0.8.0 | MIT | Copyright (c) 2022 the contributors |
| typing-inspection | 0.4.4 | MIT | Copyright (c) Pydantic Services Inc. 2025 to present |
| typing_extensions | 4.16.0 | PSF-2.0 | Copyright (c) Python Software Foundation |
| HTTPX | 0.28.1 | BSD-3-Clause | Copyright (c) 2019, Encode OSS Ltd |
| HTTP Core (httpcore) | 1.0.9 | BSD-3-Clause | Copyright (c) 2020, Encode OSS Ltd |
| h11 | 0.16.0 | MIT | Copyright (c) 2016 Nathaniel J. Smith and other contributors |
| AnyIO | 4.15.1 | MIT | Copyright (c) 2018 Alex Grönholm |
| idna | 3.20 | BSD-3-Clause | Copyright (c) 2013-2026, Kim Davies and contributors |
| certifi | 2026.7.22 | MPL-2.0 | Mozilla's CA root certificates, packaged by the certifi project. The covered file is `_internal\certifi\cacert.pem`, shipped in its source form. |

Some of these packages include further libraries:

* **ONNX Runtime** includes code from other projects (for example Abseil,
  Eigen and Protocol Buffers). Their notices are in
  `licenses\ONNX-Runtime-ThirdPartyNotices.txt`. Its GPU support is the
  `onnxruntime_providers_cuda.dll` that ships with it; the NVIDIA libraries
  that DLL needs are not part of Whisp (see the GPU runtime pack below).
* **NumPy** includes OpenBLAS (BSD-3-Clause), LAPACK (BSD-3-Clause-Open-MPI)
  and the GCC runtime library (GPL-3.0-or-later WITH GCC-exception-3.1).
  Details are in NumPy's `LICENSE.txt`.
* **python-sounddevice** includes **PortAudio** (`libportaudio64bit.dll`), by
  Ross Bencina and Phil Burk, under the MIT License.

### Microsoft Visual C++ runtime

`vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll` and related files,
copyright Microsoft Corporation. Redistributed with Whisp as Microsoft's
Visual C++ redistributable files allow.

### PyInstaller bootloader

`Whisp.exe` is built with PyInstaller. Its small start-up program (the
bootloader) is under the GNU General Public License v2.0 or later, with the
PyInstaller Bootloader Exception. The exception allows it to be used with
programs under any licence. Copyright (c) 2010-2023, PyInstaller Development
Team; Copyright (c) 2005-2009, Giovanni Bajo; based on previous work under
copyright (c) 2002 McMillan Enterprises, Inc.

### Inside Whisp's own files

| Component | Licence | Copyright | Licence file (in the app folder, under `_internal\whisp\`) |
|---|---|---|---|
| Inter typeface | SIL Open Font License 1.1 | Copyright (c) 2016 The Inter Project Authors | `assets\fonts\Inter-LICENSE.txt` |
| Silero VAD model 6.2.1 (`silero_vad.onnx`, unmodified) | MIT | Copyright (c) 2020-present Silero Team | `assets\models\SILERO_VAD_LICENSE.txt` |
| Preact | MIT | Copyright (c) 2015-present Jason Miller | `ui\main_window\web\vendor\preact\LICENSE` |
| Preact Signals (`@preact/signals`, `@preact/signals-core`) | MIT | Copyright (c) 2022-present Preact Team | `ui\main_window\web\vendor\signals\LICENSE`, `...\signals-core\LICENSE` |
| htm | Apache-2.0 | Copyright 2018 Google Inc. | `ui\main_window\web\vendor\htm\LICENSE` |
| Lucide icons | ISC (icons derived from Feather: MIT) | Copyright (c) 2026 Lucide Icons and Contributors; Feather icons copyright (c) 2013-present Cole Bemis | `ui\main_window\web\vendor\lucide\LICENSE` |

## What Whisp downloads later

Whisp downloads these only when you ask for them, straight from where their
makers publish them. Every file is pinned to an exact version and checked
against its SHA-256 before Whisp uses it. They are not part of Whisp, and
their own licences apply.

### Speech models

| Model | Licence | Credit |
|---|---|---|
| Parakeet TDT 0.6B v2 | CC-BY-4.0 | Parakeet TDT 0.6B v2 by NVIDIA (huggingface.co/nvidia/parakeet-tdt-0.6b-v2), licensed under CC BY 4.0. Converted to ONNX by Ilya Stupakov (huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx). Whisp uses the converted model without changes. |
| Parakeet TDT 0.6B v3 | CC-BY-4.0 | Parakeet TDT 0.6B v3 by NVIDIA (huggingface.co/nvidia/parakeet-tdt-0.6b-v3), licensed under CC BY 4.0. Converted to ONNX by Ilya Stupakov (huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx). Whisp uses the converted model without changes. |
| Whisper Large v3 Turbo | MIT | Whisper Large v3 Turbo by OpenAI (github.com/openai/whisper), licensed under the MIT License. ONNX conversion by the ONNX Community (huggingface.co/onnx-community/whisper-large-v3-turbo). |

CC BY 4.0: <https://creativecommons.org/licenses/by/4.0/>

### AI polish models (optional)

| Model | Licence | Source |
|---|---|---|
| Qwen3.5 4B, Qwen3.5 2B | Apache-2.0 | Qwen3.5 by the Qwen team (Alibaba Cloud). GGUF files from Unsloth (huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF, huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF). |
| Gemma 4 E4B | Apache-2.0 | Gemma 4 by Google (huggingface.co/google/gemma-4-E4B-it). GGUF files from ggml-org (huggingface.co/ggml-org/gemma-4-E4B-it-GGUF). |

### Built-in AI server (optional)

[llama.cpp](https://github.com/ggml-org/llama.cpp) build b11429
(`llama-server.exe` and its libraries), MIT License, copyright the ggml
authors. Whisp downloads the official release for your hardware from GitHub:

* The CPU build includes the LLVM OpenMP runtime (`libomp.dll`), under the
  Apache License 2.0 with LLVM Exceptions.
* The NVIDIA builds come with NVIDIA's CUDA runtime libraries (cudart and
  cuBLAS), under NVIDIA's CUDA licence.
* The Vulkan build uses the Vulkan loader that comes with your graphics driver.

### GPU runtime pack (optional, NVIDIA graphics cards only)

To run speech recognition on an NVIDIA graphics card, ONNX Runtime needs
NVIDIA's CUDA and cuDNN libraries. They take 1.6 GB on disk, so Whisp doesn't
ship them. When you choose **Turn on GPU speed** (Settings → Speech model),
Whisp downloads NVIDIA's own packages from PyPI (1.1 GB), checks them and
keeps only their DLLs:

| NVIDIA package | Version |
|---|---|
| CUDA Runtime (nvidia-cuda-runtime) | 13.4.92 |
| cuBLAS (nvidia-cublas) | 13.8.0.4 |
| cuFFT (nvidia-cufft) | 12.4.0.43 |
| cuRAND (nvidia-curand) | 10.4.4.72 |
| nvJitLink (nvidia-nvjitlink) | 13.4.92 |
| NVRTC (nvidia-cuda-nvrtc) | 13.4.92 |
| cuDNN (nvidia-cudnn-cu13) | 9.27.0.42 |

Copyright NVIDIA Corporation. They are licensed under NVIDIA's terms: the
NVIDIA CUDA Toolkit End User License Agreement and, for cuDNN, the NVIDIA
Software License Agreement with its cuDNN supplement. Each package's page on
PyPI links to its licence.
