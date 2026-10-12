"""The Whisp app window: a native Windows frame around the web UI.

The UI itself is plain HTML, CSS and JavaScript in ``web/``. It is served from
the custom ``whisp://app/`` scheme by :class:`WebSchemeHandler`, so the page
never reads the file system directly and nothing is fetched from the network.
The page talks to Python through :class:`~whisp.ui.main_window.bridge.Bridge`
over a QWebChannel.

Lifecycle: closing the window destroys it (``WA_DeleteOnClose``). The web page,
its renderer process and the in-memory browser profile go with it, so Whisp
gives that memory back while it waits in the tray. Create a new
:class:`MainWindow` to show it again; its ``destroyed`` signal says when the old
one is gone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QCoreApplication,
    QEvent,
    QFile,
    QIODevice,
    QObject,
    QSettings,
    Qt,
    QTimer,
    QUrl,
    QUrlQuery,
)
from PySide6.QtGui import QCloseEvent, QColor, QContextMenuEvent, QDesktopServices, QGuiApplication, QIcon, QPalette
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import (
    QWebEngineDownloadRequest,
    QWebEngineNewWindowRequest,
    QWebEnginePage,
    QWebEnginePermission,
    QWebEngineProfile,
    QWebEngineSettings,
    QWebEngineUrlRequestJob,
    QWebEngineUrlScheme,
    QWebEngineUrlSchemeHandler,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMenu, QVBoxLayout, QWidget

from whisp import APP_NAME
from whisp.platform.win32.system import animations_enabled, apps_use_light_theme
from whisp.platform.win32.window import style_app_window
from whisp.storage.paths import app_paths, resource_path
from whisp.ui.main_window.bridge import Bridge, BridgeAdapters
from whisp.ui.theme import ui_font

if TYPE_CHECKING:
    from whisp.app.services import AppServices

log = logging.getLogger(__name__)

SCHEME = b"whisp"
HOST = "app"
ORIGIN = "whisp://app"

DEFAULT_SIZE = (1080, 720)
MINIMUM_SIZE = (900, 600)
BACKGROUND = {"dark": "#0B0B0E", "light": "#F7F7F9"}

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}

# The two asset folders the page may read besides web/, and the file types each one serves.
_MOUNTS: dict[str, frozenset[str]] = {"fonts": frozenset({".woff2"}), "brand": frozenset({".svg"})}
_QWEBCHANNEL_JS = ":/qtwebchannel/qwebchannel.js"

_scheme_registered = False


def register_web_scheme() -> None:
    """Register the ``whisp://`` URL scheme with Qt WebEngine.

    Must run before the QApplication is created. Calling it again is harmless.
    """
    global _scheme_registered
    if _scheme_registered:
        return
    if QCoreApplication.instance() is not None:
        raise RuntimeError("register_web_scheme() must be called before the QApplication is created")
    scheme = QWebEngineUrlScheme(SCHEME)
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    scheme.setFlags(
        QWebEngineUrlScheme.Flag.SecureScheme
        | QWebEngineUrlScheme.Flag.LocalScheme
        | QWebEngineUrlScheme.Flag.LocalAccessAllowed
        | QWebEngineUrlScheme.Flag.CorsEnabled
    )
    QWebEngineUrlScheme.registerScheme(scheme)
    # Qt WebEngine shares OpenGL contexts with the widget stack; this must be set before the app exists.
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    _scheme_registered = True


@dataclass(frozen=True)
class WebResponse:
    status: int
    """200 (found), 403 (refused: unsafe path) or 404 (no such file)."""
    mime: str = ""
    body: bytes = b""


class WebResources:
    """Maps ``whisp://app/<path>`` to file contents. Pure, so it is easy to test.

    ``/fonts/*.woff2`` and ``/brand/*.svg`` come from the shared assets folder,
    ``/qwebchannel.js`` from Qt's own resources, and everything else from ``web/``.
    """

    def __init__(self, web_root: Path, assets_root: Path) -> None:
        self._web_root = web_root.resolve()
        self._assets_root = assets_root.resolve()

    @classmethod
    def bundled(cls) -> WebResources:
        return cls(resource_path("ui", "main_window", "web"), resource_path("assets"))

    def get(self, path: str) -> WebResponse:
        parts = _safe_parts(path)
        if parts is None:
            return WebResponse(403)
        if not parts:
            parts = ["index.html"]
        if parts == ["qwebchannel.js"]:
            return _qt_resource(_QWEBCHANNEL_JS, MIME_TYPES[".js"])
        allowed = _MOUNTS.get(parts[0])
        if allowed is not None:
            root, relative = (self._assets_root / parts[0]).resolve(), parts[1:]
        else:
            root, relative, allowed = self._web_root, parts, frozenset(MIME_TYPES)
        if not relative:
            return WebResponse(404)
        try:
            resolved = root.joinpath(*relative).resolve()
        except OSError:
            return WebResponse(403)
        if not resolved.is_relative_to(root):
            return WebResponse(403)
        suffix = resolved.suffix.lower()
        if suffix not in allowed or suffix not in MIME_TYPES or not resolved.is_file():
            return WebResponse(404)
        try:
            body = resolved.read_bytes()
        except OSError:
            log.warning("Could not read web resource %s", "/".join(parts))
            return WebResponse(404)
        return WebResponse(200, MIME_TYPES[suffix], body)


def _safe_parts(path: str) -> list[str] | None:
    """Split a decoded URL path into segments, or None if it tries to leave its folder."""
    path = path or "/"
    if not path.startswith("/") or "\\" in path or "\x00" in path:
        return None
    parts = [p for p in path.split("/") if p]
    for part in parts:
        if part in (".", "..") or part.startswith(".") or ":" in part or part != part.rstrip(". "):
            return None
    return parts


def _qt_resource(name: str, mime: str) -> WebResponse:
    file = QFile(name)
    if not file.open(QIODevice.OpenModeFlag.ReadOnly):
        return WebResponse(404)
    try:
        return WebResponse(200, mime, file.readAll().data())
    finally:
        file.close()


class WebSchemeHandler(QWebEngineUrlSchemeHandler):
    """Serves ``whisp://app/...`` from :class:`WebResources`."""

    def __init__(self, resources: WebResources, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._resources = resources

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:
        url = job.requestUrl()
        if job.requestMethod().data() != b"GET" or url.host() != HOST:
            job.fail(QWebEngineUrlRequestJob.Error.RequestDenied)
            return
        response = self._resources.get(url.path(QUrl.ComponentFormattingOption.FullyDecoded))
        if response.status == 403:
            log.warning("Refused an unsafe web UI path")
            job.fail(QWebEngineUrlRequestJob.Error.RequestDenied)
            return
        if response.status != 200:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        buffer = QBuffer(job)  # the job owns and frees it
        buffer.setData(QByteArray(response.body))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        job.setAdditionalResponseHeaders(
            {
                QByteArray(b"X-Content-Type-Options"): QByteArray(b"nosniff"),
                QByteArray(b"Cache-Control"): QByteArray(b"no-store"),
            }
        )
        job.reply(response.mime.encode("ascii"), buffer)


class _AppPage(QWebEnginePage):
    """Keeps the page on ``whisp://app``; web links open in the default browser."""

    def acceptNavigationRequest(
        self, url: QUrl | str, navigation_type: QWebEnginePage.NavigationType, is_main_frame: bool
    ) -> bool:
        url = QUrl(url)
        if url.scheme() == SCHEME.decode() and url.host() == HOST:
            return True
        if url.scheme() in ("http", "https"):
            if navigation_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
                QDesktopServices.openUrl(url)
            return False
        log.warning("Blocked the web UI from navigating to a %s: URL", url.scheme() or "relative")
        return False

    def javaScriptConsoleMessage(
        self, level: QWebEnginePage.JavaScriptConsoleMessageLevel, message: str, line: int, source: str
    ) -> None:
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel:
            return
        log.warning("Web UI: %s (%s:%d)", message[:300], QUrl(source).path(), line)


class _WebView(QWebEngineView):
    """Shows a context menu only in text fields, with editing actions only."""

    _EDIT_ACTIONS = (
        QWebEnginePage.WebAction.Undo,
        QWebEnginePage.WebAction.Redo,
        None,
        QWebEnginePage.WebAction.Cut,
        QWebEnginePage.WebAction.Copy,
        QWebEnginePage.WebAction.Paste,
        None,
        QWebEnginePage.WebAction.SelectAll,
    )

    def __init__(self, parent: QWidget, menu_stylesheet: Callable[[], str]) -> None:
        super().__init__(parent)
        self._menu_stylesheet = menu_stylesheet

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        request = self.lastContextMenuRequest()
        event.accept()
        if request is None or not request.isContentEditable():
            return
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        menu.setWindowFlags(menu.windowFlags() | Qt.WindowType.FramelessWindowHint)
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        menu.setFont(ui_font(13))
        menu.setStyleSheet(self._menu_stylesheet())
        for web_action in self._EDIT_ACTIONS:
            if web_action is None:
                menu.addSeparator()
            else:
                menu.addAction(self.pageAction(web_action))
        menu.popup(event.globalPos())


class MainWindow(QWidget):
    """The Whisp window: Home, History, Dictionary, Snippets, Styles, Settings and onboarding."""

    def __init__(self, services: AppServices, *, adapters: BridgeAdapters | None = None) -> None:
        super().__init__()
        self._services = services
        self._torn_down = False
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(str(resource_path("assets", "brand", "logo.svg"))))
        self._theme = self._effective_theme()
        self._reduced_motion = not animations_enabled()

        # Creation order matters for teardown: Qt deletes children in this order, so the
        # view (and the page it owns) always goes before the profile the page runs in.
        self._view = _WebView(self, self._menu_stylesheet)
        self._profile = _create_profile(self)
        self._page = _AppPage(self._profile, self._view)
        self._view.setPage(self._page)
        self._bridge = Bridge(services, adapters, theme=self._theme, reduced_motion=self._reduced_motion, parent=self)
        self._channel = QWebChannel(self._page)
        self._channel.registerObject("bridge", self._bridge)
        self._page.setWebChannel(self._channel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view)

        self._page.renderProcessTerminated.connect(self._on_renderer_gone)
        self._page.loadFinished.connect(self._on_load_finished)
        self._page.newWindowRequested.connect(_open_in_browser)
        self._page.permissionRequested.connect(_deny_permission)
        self._profile.downloadRequested.connect(_refuse_download)
        services.settings.changed.connect(self._on_settings_changed)
        QGuiApplication.styleHints().colorSchemeChanged.connect(self._refresh_theme)

        self._apply_theme(self._theme, force=True)
        self._restore_geometry()
        self._view.load(self._start_url())

    # -- public API -----------------------------------------------------------------

    def show_page(self, page: str, section: str | None = None) -> None:
        """Bring the window forward on ``page``.

        ``page`` is one of "home", "history", "dictionary", "snippets", "styles",
        "settings" or "onboarding". ``section`` optionally picks a Settings tab
        (general, shortcuts, microphone, model, typing, overlay, privacy, about).
        """
        self._bridge.navigate(page, section)
        if self.isMinimized():
            self.showNormal()
        elif not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    # -- Qt events ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_geometry()
        self._teardown()
        event.accept()

    def changeEvent(self, event: QEvent) -> None:
        # The window can be reopened long after Windows' theme or animation setting
        # changed; re-check whenever it becomes active.
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow() and not self._torn_down:
            self._refresh_theme()
            reduced = not animations_enabled()
            if reduced != self._reduced_motion:
                self._reduced_motion = reduced
                self._bridge.set_reduced_motion(reduced)
        super().changeEvent(event)

    # -- internals ------------------------------------------------------------------

    def _teardown(self) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        self._bridge.shutdown()
        for signal, slot in (
            (self._services.settings.changed, self._on_settings_changed),
            (QGuiApplication.styleHints().colorSchemeChanged, self._refresh_theme),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                log.debug("Window signal was already disconnected")
        self._channel.deregisterObject(self._bridge)
        self._view.stop()

    def _start_url(self) -> QUrl:
        url = QUrl(f"{ORIGIN}/index.html")
        query = QUrlQuery()
        query.addQueryItem("theme", self._theme)
        query.addQueryItem("motion", "reduced" if self._reduced_motion else "full")
        url.setQuery(query)
        return url

    def _effective_theme(self) -> str:
        choice = self._services.settings.current.general.theme
        if choice == "system":
            return "light" if apps_use_light_theme() else "dark"
        return choice

    def _refresh_theme(self) -> None:
        if not self._torn_down:
            self._apply_theme(self._effective_theme())

    def _apply_theme(self, theme: str, *, force: bool = False) -> None:
        if theme == self._theme and not force:
            return
        self._theme = theme
        background = QColor(BACKGROUND[theme])
        try:
            style_app_window(self, dark=theme == "dark")
        except OSError:
            log.warning("Could not colour the window's title bar", exc_info=True)
        self._page.setBackgroundColor(background)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, background)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self._bridge.set_theme(theme)

    def _on_settings_changed(self, sections: object) -> None:
        if isinstance(sections, (set, frozenset, list, tuple)) and "general" in sections:
            self._refresh_theme()

    def _menu_stylesheet(self) -> str:
        # The web UI's surface-2, text-1, text-3, surface-3 and border-strong tokens.
        if self._theme == "dark":
            fill, text, muted, hover, border = "#18181E", "#F4F4F6", "#82828D", "#25252F", "rgba(255,255,255,0.12)"
        else:
            fill, text, muted, hover, border = "#FFFFFF", "#0E0E12", "#6B6B77", "#EEEEF3", "rgba(12,12,20,0.14)"
        return (
            f"QMenu {{ background: {fill}; color: {text}; border: 1px solid {border}; border-radius: 8px;"
            " padding: 4px; }"
            " QMenu::item { padding: 6px 28px 6px 12px; border-radius: 6px; }"
            f" QMenu::item:selected {{ background: {hover}; }}"
            f" QMenu::item:disabled {{ color: {muted}; }}"
            f" QMenu::separator {{ height: 1px; background: {border}; margin: 4px 6px; }}"
        )

    def _on_renderer_gone(self, status: QWebEnginePage.RenderProcessTerminationStatus, exit_code: int) -> None:
        if status == QWebEnginePage.RenderProcessTerminationStatus.NormalTerminationStatus or self._torn_down:
            return
        log.warning("The window's web renderer stopped (%s, code %d); reloading", status.name, exit_code)
        self._bridge.page_reloaded()
        QTimer.singleShot(300, self._reload)

    def _reload(self) -> None:
        if not self._torn_down:
            self._view.load(self._start_url())

    def _on_load_finished(self, ok: bool) -> None:
        if not ok and not self._torn_down:
            log.error("The window's web UI failed to load")

    def _geometry_store(self) -> QSettings:
        return QSettings(str(app_paths().config_dir / "window.ini"), QSettings.Format.IniFormat)

    def _restore_geometry(self) -> None:
        screen = QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        min_w, min_h = MINIMUM_SIZE
        width, height = DEFAULT_SIZE
        if available is not None:
            min_w, min_h = min(min_w, available.width()), min(min_h, available.height())
            width, height = min(width, available.width()), min(height, available.height())
        self.setMinimumSize(min_w, min_h)
        saved = self._geometry_store().value("main/geometry")
        if isinstance(saved, QByteArray) and not saved.isEmpty() and self.restoreGeometry(saved):
            return
        self.resize(width, height)
        if available is not None:
            frame = self.frameGeometry()
            frame.moveCenter(available.center())
            self.move(frame.topLeft())

    def _save_geometry(self) -> None:
        store = self._geometry_store()
        store.setValue("main/geometry", self.saveGeometry())
        store.sync()
        if store.status() != QSettings.Status.NoError:
            log.warning("Could not save the window's size and position")


def _create_profile(parent: QObject) -> QWebEngineProfile:
    """An off-the-record profile: no cache, cookies or storage ever reach the disk."""
    profile = QWebEngineProfile(parent)
    profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
    profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies)
    profile.setSpellCheckEnabled(False)
    attribute = QWebEngineSettings.WebAttribute
    settings = profile.settings()
    for name, enabled in (
        (attribute.JavascriptCanOpenWindows, False),
        (attribute.JavascriptCanAccessClipboard, False),
        (attribute.JavascriptCanPaste, False),
        (attribute.LocalContentCanAccessRemoteUrls, False),
        (attribute.LocalContentCanAccessFileUrls, False),
        (attribute.AllowWindowActivationFromJavaScript, False),
        (attribute.PluginsEnabled, False),
        (attribute.PdfViewerEnabled, False),
        (attribute.FullScreenSupportEnabled, False),
        (attribute.ScreenCaptureEnabled, False),
        (attribute.WebGLEnabled, False),
        (attribute.NavigateOnDropEnabled, False),
        (attribute.ErrorPageEnabled, False),
        (attribute.HyperlinkAuditingEnabled, False),
        (attribute.DnsPrefetchEnabled, False),
        (attribute.AutoLoadIconsForPage, False),
        (attribute.TouchIconsEnabled, False),
        (attribute.ScrollAnimatorEnabled, True),
        (attribute.FocusOnNavigationEnabled, True),
    ):
        settings.setAttribute(name, enabled)
    handler = WebSchemeHandler(WebResources.bundled(), profile)
    profile.installUrlSchemeHandler(SCHEME, handler)
    return profile


def _open_in_browser(request: QWebEngineNewWindowRequest) -> None:
    url = request.requestedUrl()
    if url.scheme() in ("http", "https"):
        QDesktopServices.openUrl(url)


def _deny_permission(permission: QWebEnginePermission) -> None:
    permission.deny()


def _refuse_download(download: QWebEngineDownloadRequest) -> None:
    download.cancel()
