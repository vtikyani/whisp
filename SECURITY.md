# Security

## Reporting a problem

Please don't report security problems in a public issue.

Report them privately through GitHub instead: open this repository's
**Security** tab and choose **Report a vulnerability**. Only the maintainers
can see the report.

Please include:

- what the problem is and what an attacker could do with it;
- the steps to reproduce it;
- your Whisp version (`Whisp.exe --version`) and Windows version.

Leave out anything personal: no recordings, transcripts or unedited logs.

We fix security problems in the latest release. The fix ships as a new
installer.

Problems in a library Whisp uses (Qt, ONNX Runtime, llama.cpp and so on)
belong with that project. Tell us too if Whisp needs to update to a fixed
version.

## What Whisp does with your data

Whisp turns speech into text on your own PC. Here's what that means in
practice.

**Your voice stays on your PC.** Speech recognition, text cleanup and the
built-in AI polish all run locally. Whisp has no account, no telemetry and
no automatic update check.

**What it stores:**

| What | Where |
|---|---|
| Settings, dictionary and snippets | `%APPDATA%\Whisp` |
| Dictation history (optional) | `%LOCALAPPDATA%\Whisp\history.db` |
| Downloaded models and runtimes | `%LOCALAPPDATA%\Whisp\models` and `runtime` |
| Logs | `%LOCALAPPDATA%\Whisp\logs` |

- History is kept for 30 days by default. You can shorten or lengthen that,
  clear it, or turn it off. Deleted entries are overwritten on disk.
- Logs never contain what you said, your selections, your clipboard or window
  titles.
- An API key for your own AI server is encrypted with Windows DPAPI, so only
  your Windows account can read it.
- Text Whisp puts on the clipboard to paste is kept out of Windows clipboard
  history and cloud clipboard. Whisp puts back what was there before.

**When it goes online:** only to download what you ask for. Speech and AI
models come from Hugging Face, the GPU runtime pack from NVIDIA's packages on
PyPI, and the built-in AI server from llama.cpp's releases on GitHub. Each
file is pinned to an exact version and its SHA-256 is checked before Whisp
uses it.

**The built-in AI server** runs as a separate process. It listens only on
`127.0.0.1` and needs a random key that changes every time it starts, so other
programs and web pages can't use it. It's tied to Whisp and stops when Whisp
stops, even if Whisp crashes.

**Your own AI server:** if you set up Ollama or an OpenAI-compatible server
for AI polish, Whisp sends your dictated text to it. If that server is on
another computer or in the cloud, your text leaves your PC.

**The window's web page** is served by Whisp itself, from files inside the
app. It can't load anything from the internet, has a strict Content Security
Policy, and can only open web links in your browser.

**Installing** needs no administrator rights. Whisp installs for your user
account only and changes nothing outside it.
