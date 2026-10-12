import { useEffect, useRef } from "preact/hooks";
import { html } from "./html.js";
import { Icon } from "./icons.js";
import { DictionaryPage } from "./pages/dictionary.js";
import { HistoryPage } from "./pages/history.js";
import { HomePage } from "./pages/home.js";
import { OnboardingPage } from "./pages/onboarding.js";
import { SettingsPage } from "./pages/settings.js";
import { SnippetsPage } from "./pages/snippets.js";
import { StylesPage } from "./pages/styles.js";
import { engine, navigate, route } from "./store.js";
import { DialogHost, Toaster } from "./ui.js";

const NAV = [
  { page: "home", label: "Home", icon: "house" },
  { page: "history", label: "History", icon: "history" },
  { page: "dictionary", label: "Dictionary", icon: "book-a" },
  { page: "snippets", label: "Snippets", icon: "text-quote" },
  { page: "styles", label: "Styles", icon: "case-sensitive" },
  { page: "settings", label: "Settings", icon: "settings" },
];

const PAGES = {
  home: HomePage,
  history: HistoryPage,
  dictionary: DictionaryPage,
  snippets: SnippetsPage,
  styles: StylesPage,
  settings: SettingsPage,
  onboarding: OnboardingPage,
};

const ITEM_PITCH = 36; // 34 px item + 2 px gap
const MARKER_OFFSET = 9; // centres the 16 px marker on a 34 px item

function shortModel(status) {
  const name = status.model_name || status.model_id || "Speech model";
  return name.split(/\s+/)[0];
}

function shortDevice(status) {
  if (status.device !== "cuda") return "CPU";
  return (status.device_name || "GPU").replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
}

function describeEngine(status) {
  if (!status) return { dot: "", title: "Speech engine", sub: "Starting…" };
  switch (status.state) {
    case "ready":
      return { dot: "dot-success", title: `${shortModel(status)} · ${shortDevice(status)}`, sub: "Ready" };
    case "loading":
      return { dot: "dot-warning dot-pulse", title: shortModel(status), sub: status.message || "Loading the model…" };
    case "unloaded":
      return { dot: "dot-success", title: `${shortModel(status)} · ${shortDevice(status)}`, sub: "Loads when you dictate" };
    case "not_installed":
      return { dot: "dot-danger", title: "No speech model", sub: "Download one to start" };
    default:
      return { dot: "dot-danger", title: "Speech engine stopped", sub: status.message || "Open settings to fix it" };
  }
}

function EngineChip() {
  const status = engine.value;
  const { dot, title, sub } = describeEngine(status);
  const full = status?.model_name ? `${status.model_name} on ${status.device_name}` : title;
  return html`<button
    class="engine-chip"
    title=${full}
    aria-label=${`Speech engine: ${title}. ${sub}. Open speech model settings.`}
    onClick=${() => navigate("settings", "model")}
  >
    <span class=${`dot ${dot}`}></span>
    <span class="engine-chip-text">
      <span class="engine-chip-title">${title}</span>
      <span class="engine-chip-sub">${sub}</span>
    </span>
    <${Icon} name="chevron-right" size=${14} />
  </button>`;
}

function Sidebar() {
  const current = route.value.page;
  const index = NAV.findIndex((item) => item.page === current);
  return html`<aside class="sidebar">
    <div class="brand">
      <img src="/brand/logo.svg" alt="" />
      <span class="brand-name">Whisp</span>
    </div>
    <nav class="nav" aria-label="Main">
      ${index >= 0 && html`<span class="nav-marker" style=${{ "--y": `${index * ITEM_PITCH + MARKER_OFFSET}px` }}></span>`}
      ${NAV.map(
        (item) => html`<button
          class="nav-item"
          aria-current=${item.page === current ? "page" : undefined}
          onClick=${() => navigate(item.page)}
        >
          <${Icon} name=${item.icon} size=${18} />${item.label}
        </button>`,
      )}
    </nav>
    <div class="sidebar-spacer"></div>
    <${EngineChip} />
  </aside>`;
}

export function App() {
  const { page, section } = route.value;
  const onboarding = page === "onboarding";
  const main = useRef(null);
  const Page = PAGES[page];

  useEffect(() => {
    main.current?.scrollTo(0, 0);
  }, [page, section]);

  return html`<div class=${`app ${onboarding ? "is-onboarding" : "has-sidebar"}`}>
    ${!onboarding && html`<${Sidebar} />`}
    <main class="main" ref=${main}>
      <${Page} key=${page} />
    </main>
    <${Toaster} />
    <${DialogHost} />
  </div>`;
}
