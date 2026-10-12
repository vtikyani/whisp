import os
import sys
import logging

log = logging.getLogger("whisp")

STARTUP_DIR = os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs\Startup")
SHORTCUT_NAME = "Whisp.lnk"


def _get_shortcut_path():
    return os.path.join(STARTUP_DIR, SHORTCUT_NAME)


def _get_target():
    """Get the pythonw.exe path and script path for the shortcut."""
    python_dir = os.path.dirname(sys.executable)
    pythonw = os.path.join(python_dir, "pythonw.exe")
    if not os.path.exists(pythonw):
        pythonw = sys.executable  # fallback to python.exe
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run.pyw")
    return pythonw, script


def enable_autostart():
    """Create a .lnk shortcut in shell:startup using PowerShell."""
    pythonw, script = _get_target()
    shortcut_path = _get_shortcut_path()
    working_dir = os.path.dirname(script)

    ps_script = f"""
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut('{shortcut_path}')
$sc.TargetPath = '{pythonw}'
$sc.Arguments = '"{script}"'
$sc.WorkingDirectory = '{working_dir}'
$sc.Description = 'Whisp Voice-to-Text'
$sc.Save()
"""
    os.system(f'powershell -Command "{ps_script.strip()}"')
    log.info("Autostart enabled: %s", shortcut_path)


def disable_autostart():
    """Remove the startup shortcut."""
    shortcut_path = _get_shortcut_path()
    if os.path.exists(shortcut_path):
        os.remove(shortcut_path)
        log.info("Autostart disabled")


def is_autostart_enabled():
    return os.path.exists(_get_shortcut_path())
