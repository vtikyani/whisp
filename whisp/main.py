import os
import sys
import logging

# IMPORTANT: onnxruntime must be imported before PyQt6 to avoid DLL conflicts on Windows
import onnxruntime  # noqa: F401

from whisp.config import load_config, CONFIG_DIR

LOG_PATH = os.path.join(CONFIG_DIR, "whisp.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("whisp")


def main():
    config = load_config()
    log.info("Whisp starting with config: %s", config)

    # Handle autostart
    if config.get("auto_start", True):
        try:
            from whisp.autostart import enable_autostart, is_autostart_enabled

            if not is_autostart_enabled():
                enable_autostart()
        except Exception as e:
            log.warning("Could not set up autostart: %s", e)

    from whisp.app import WhispApp

    app = WhispApp(config)
    sys.exit(app.run())


if __name__ == "__main__":
    main()
