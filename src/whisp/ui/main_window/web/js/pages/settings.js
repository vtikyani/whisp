import { useRef, useState } from "preact/hooks";
import { html } from "../html.js";
import { call } from "../bridge.js";
import { languageName, plural } from "../format.js";
import { useEvent, useResource } from "../hooks.js";
import { Icon } from "../icons.js";
import { DeviceSelect, LevelBar, useDevices, useMicMeter } from "../mic.js";
import { GpuCard, ModelCard, catalogError, useModels } from "../models.js";
import { AppOverrides } from "../overrides.js";
import { ShortcutRecorder } from "../shortcut.js";
import { appInfo, navigate, reportError, route, saveSettings, settings, toast } from "../store.js";
import {
  Badge,
  Button,
  Card,
  IconButton,
  Notice,
  Row,
  Segmented,
  Select,
  Slider,
  Spinner,
  Toggle,
  confirmDialog,
} from "../ui.js";

export const SECTIONS = [
  { id: "general", label: "General" },
  { id: "shortcuts", label: "Shortcuts" },
  { id: "microphone", label: "Microphone" },
  { id: "model", label: "Speech model" },
  { id: "typing", label: "Typing" },
  { id: "overlay", label: "Overlay" },
  { id: "privacy", label: "Privacy" },
  { id: "about", label: "About" },
];

const openUrl = (url) => call("app.open_url", { url }).catch(reportError);
const openFolder = (name) => call("app.open_folder", { name }).catch(reportError);
const openWindowsSettings = (page) => call("app.open_system_settings", { page }).catch(reportError);

// General ------------------------------------------------------------------------------------

function GeneralSection() {
  const general = settings.value.general;
  const autostart = useResource(() => call("autostart.status"), []);
  const status = autostart.data?.status;

  const setAutostart = async (enabled) => {
    try {
      await call("autostart.set", { enabled });
    } catch (error) {
      reportError(error);
    }
    autostart.reload();
  };

  const resetAll = async () => {
    const ok = await confirmDialog({
      title: "Reset all settings?",
      text: "Shortcuts, styles, your dictionary, snippets and every other setting go back to how they were when you installed Whisp. Your history is kept.",
      confirmLabel: "Reset everything",
      danger: true,
    });
    if (!ok) return;
    try {
      settings.value = await call("settings.reset", {});
      toast("Settings reset.");
    } catch (error) {
      reportError(error);
    }
  };

  return html`<div class="stack">
    <${Card}>
      <div class="rows">
        <${Row} label="Launch at startup" hint="Start Whisp when you sign in to Windows, so it's always ready.">
          <${Toggle}
            label="Launch at startup"
            checked=${status === "enabled"}
            disabled=${!status || status === "blocked"}
            onChange=${setAutostart}
          />
        <//>
        ${status === "blocked" &&
        html`<div class="row-sub">
          <${Notice}
            kind="warning"
            title="Windows has turned this off"
            actions=${html`<${Button} kind="secondary" size="sm" onClick=${() => openWindowsSettings("startup")}>Open Startup apps<//>`}
          >
            Turn Whisp back on in Windows Settings → Apps → Startup.
          <//>
        </div>`}
        <${Row} label="Appearance" hint="System follows your Windows light or dark mode.">
          <${Segmented}
            label="Appearance"
            value=${general.theme}
            options=${[
              { value: "system", label: "System", icon: "monitor" },
              { value: "light", label: "Light", icon: "sun" },
              { value: "dark", label: "Dark", icon: "moon" },
            ]}
            onChange=${(theme) => saveSettings({ general: { theme } })}
          />
        <//>
        <${Row} label="Notifications" hint="Let Whisp tell you when something needs attention, like a missing microphone.">
          <${Toggle}
            label="Notifications"
            checked=${general.show_notifications}
            onChange=${(on) => saveSettings({ general: { show_notifications: on } })}
          />
        <//>
      </div>
    <//>
    <${Card}>
      <div class="rows">
        <${Row} label="Setup guide" hint="Choose your microphone, speech model and shortcut again, step by step.">
          <${Button} kind="secondary" onClick=${() => navigate("onboarding")}>Run setup<//>
        <//>
        <${Row} label="Reset all settings" hint="Go back to the defaults. Your history stays.">
          <${Button} kind="danger-ghost" onClick=${resetAll}>Reset<//>
        <//>
      </div>
    <//>
  </div>`;
}

// Shortcuts ------------------------------------------------------------------------------------

// Labels match the tray's notifications (app.controller.ACTION_LABELS).
const ACTIONS = [
  { id: "dictate", label: "Dictation", hint: "Hold to talk, let go to insert. A quick tap keeps listening hands-free." },
  {
    id: "command",
    label: "Command mode",
    hint: "Select some text, hold, and say how to change it. Whisp copies your selection with Ctrl + C, so Windows clipboard history (Win + V) may keep it.",
    clearable: true,
  },
  { id: "paste_last", label: "Paste last transcript", hint: "Types your most recent dictation again.", clearable: true },
  { id: "copy_last", label: "Copy last transcript", hint: "Puts your most recent dictation on the clipboard.", clearable: true },
  { id: "hands_free", label: "Hands-free", hint: "Press once to start listening and again to stop.", clearable: true },
];

function ShortcutsSection() {
  const hotkeys = settings.value.hotkeys;
  const polishOn = settings.value.polish.enabled;
  const initial = useResource(() => call("hotkeys.errors"), []);
  const [live, setLive] = useState(null);
  useEvent("hotkeys.errors", setLive);
  const errors = live || initial.data || {};

  const reset = async () => {
    const ok = await confirmDialog({
      title: "Reset shortcuts?",
      text: "All shortcuts and the tap timing go back to their defaults.",
      confirmLabel: "Reset shortcuts",
    });
    if (!ok) return;
    try {
      settings.value = await call("settings.reset", { section: "hotkeys" });
      toast("Shortcuts reset.");
    } catch (error) {
      reportError(error);
    }
  };

  return html`<div class="stack">
    <${Card}>
      <div class="rows">
        ${ACTIONS.map(
          (action) => html`<${Row}
            key=${action.id}
            top
            label=${action.label}
            hint=${html`${action.hint}${action.id === "command" && !polishOn
              ? html` <button class="link-btn" onClick=${() => navigate("styles")}>Turn on AI polish</button> to use it.`
              : ""}`}
          >
            <${ShortcutRecorder}
              action=${action.id}
              chord=${hotkeys[action.id]}
              error=${errors[action.id]}
              onSave=${(chord) => saveSettings({ hotkeys: { [action.id]: chord } })}
              onClear=${action.clearable ? () => saveSettings({ hotkeys: { [action.id]: "" } }) : null}
              changeLabel=${hotkeys[action.id] ? "Change" : "Set"}
            />
          <//>`,
        )}
      </div>
    <//>
    <${Card}>
      <div class="rows">
        <${Row} label="Quick tap" hint=${`A press shorter than this starts hands-free mode instead of hold-to-talk.`}>
          <${Slider}
            label="Quick tap length"
            min=${150}
            max=${1000}
            step=${10}
            value=${hotkeys.tap_to_lock_ms}
            format=${(value) => `${value} ms`}
            onCommit=${(value) => saveSettings({ hotkeys: { tap_to_lock_ms: value } })}
          />
        <//>
        <${Row} label="Reset shortcuts" hint="Go back to the shortcuts Whisp started with.">
          <${Button} kind="ghost" icon="rotate-ccw" onClick=${reset}>Reset<//>
        <//>
      </div>
    <//>
  </div>`;
}

// Microphone --------------------------------------------------------------------------------

function meterMessage(meter) {
  if (meter.state === "started") return `Listening on ${meter.device_name}. Say something to test it.`;
  if (meter.state === "error") return meter.message;
  if (meter.state === "stopped") return "The microphone closed.";
  return "Opening the microphone…";
}

export function MicrophoneMeter({ device }) {
  const meter = useMicMeter(device);
  return html`<div class="meter-block">
    <div class="meter-line">
      <${Icon} name=${meter.state === "error" ? "mic-off" : "mic"} size=${16} />
      <${LevelBar} level=${meter.level} />
    </div>
    <p class=${`meter-status${meter.state === "error" ? " is-error" : ""}`} role="status">
      ${meterMessage(meter)}
      ${meter.state === "error" &&
      html`<span class="meter-actions">
        <button class="link-btn" onClick=${meter.retry}>Try again</button>
        <button class="link-btn" onClick=${() => openWindowsSettings("microphone")}>Check microphone access</button>
      </span>`}
    </p>
  </div>`;
}

const RECORDING_LIMITS = [1, 2, 5, 10, 15, 20, 30, 45, 60];

function MicrophoneSection() {
  const audio = settings.value.audio;
  const devices = useDevices();
  const limits = RECORDING_LIMITS.includes(audio.max_recording_minutes)
    ? RECORDING_LIMITS
    : [...RECORDING_LIMITS, audio.max_recording_minutes].sort((a, b) => a - b);
  return html`<div class="stack">
    <${Card}>
      <div class="rows">
        <${Row} label="Microphone" hint="Whisp listens to this device when you dictate.">
          <${DeviceSelect}
            devices=${devices.data?.devices}
            value=${audio.input_device}
            onChange=${(device) => saveSettings({ audio: { input_device: device } })}
          />
          <${IconButton} icon="refresh-cw" label="Look for microphones again" tipSide="left" onClick=${devices.reload} />
        <//>
        <div class="row is-block">
          <${MicrophoneMeter} device=${audio.input_device} />
        </div>
      </div>
      ${devices.error && html`<div class="row-sub"><${Notice} kind="danger" title="Couldn't list microphones">${devices.error.message}<//></div>`}
    <//>
    <${Card}>
      <div class="rows">
        <${Row} label="Sounds" hint="A soft tone when Whisp starts and stops listening.">
          <${Toggle} label="Sounds" checked=${audio.sounds_enabled} onChange=${(on) => saveSettings({ audio: { sounds_enabled: on } })} />
        <//>
        <${Row} label="Volume">
          <${Slider}
            label="Sound volume"
            min=${0}
            max=${100}
            step=${5}
            value=${Math.round(audio.sounds_volume * 100)}
            format=${(value) => `${value}%`}
            onCommit=${(value) => saveSettings({ audio: { sounds_volume: value / 100 } })}
          />
        <//>
        <${Row}
          label="Mute other apps while dictating"
          hint="Silences music and videos while you talk, so Whisp hears you clearly. Calls and apps using a microphone stay on."
        >
          <${Toggle}
            label="Mute other apps while dictating"
            checked=${audio.mute_other_apps}
            onChange=${(on) => saveSettings({ audio: { mute_other_apps: on } })}
          />
        <//>
        <${Row} label="Longest recording" hint="Whisp stops listening after this long, so a forgotten hands-free session can't run forever.">
          <${Select}
            label="Longest recording"
            value=${audio.max_recording_minutes}
            options=${limits.map((minutes) => ({ value: minutes, label: plural(minutes, "minute") }))}
            onChange=${(value) => saveSettings({ audio: { max_recording_minutes: Number(value) } })}
          />
        <//>
      </div>
    <//>
  </div>`;
}

// Speech model ---------------------------------------------------------------------------------

const UNLOAD_CHOICES = [0, 5, 15, 30, 60];

function ModelSection() {
  const data = useModels();
  const t = settings.value.transcription;
  if (!data) {
    if (catalogError.value) {
      return html`<${Notice} kind="danger" title="Couldn't load the speech models">${catalogError.value.message}<//>`;
    }
    return html`<div class="panel-loading"><${Spinner} /> Checking your models and graphics card…</div>`;
  }
  const selected = data.models.find((model) => model.id === data.selected);
  const multilingual = selected && selected.languages.length > 1;
  const languageOptions = multilingual
    ? [
        { value: "auto", label: "Detect automatically" },
        ...selected.languages
          .map((code) => ({ value: code, label: languageName(code) }))
          .sort((a, b) => a.label.localeCompare(b.label)),
      ]
    : [{ value: t.language, label: selected ? languageName(selected.languages[0]) : "Detect automatically" }];
  const unloadChoices = UNLOAD_CHOICES.includes(t.unload_after_minutes)
    ? UNLOAD_CHOICES
    : [...UNLOAD_CHOICES, t.unload_after_minutes].sort((a, b) => a - b);

  return html`<div class="stack">
    <${GpuCard} gpu=${data.gpu} />
    <div class="models">${data.models.map((model) => html`<${ModelCard} key=${model.id} model=${model} data=${data} />`)}</div>
    <${Card} title="Recognition">
      <div class="rows">
        <${Row} label="Run on" hint="Auto uses your NVIDIA graphics card when it can, and your processor otherwise.">
          <${Segmented}
            label="Run speech recognition on"
            value=${t.device}
            options=${[
              { value: "auto", label: "Auto" },
              { value: "gpu", label: "GPU", disabled: !data.gpu.cuda_ready, title: data.gpu.cuda_ready ? undefined : data.gpu.detail },
              { value: "cpu", label: "CPU" },
            ]}
            onChange=${(device) => saveSettings({ transcription: { device } })}
          />
        <//>
        <${Row}
          label="Language"
          hint=${multilingual ? "Pick one if you always speak the same language. It's a little faster and more accurate." : "This model understands English only."}
        >
          <${Select}
            label="Language"
            value=${t.language}
            options=${languageOptions}
            disabled=${!multilingual}
            onChange=${(language) => saveSettings({ transcription: { language } })}
          />
        <//>
        <${Row} label="Keep the model ready" hint="Keeping it in memory means every dictation starts instantly.">
          <${Select}
            label="Keep the model ready"
            value=${t.unload_after_minutes}
            options=${unloadChoices.map((minutes) => ({
              value: minutes,
              label: minutes === 0 ? "Always" : `For ${plural(minutes, "minute")} after use`,
            }))}
            onChange=${(value) => saveSettings({ transcription: { unload_after_minutes: Number(value) } })}
          />
        <//>
        <${Row} label="Live preview" hint="Show your words in the overlay while you speak.">
          <${Toggle} label="Live preview" checked=${t.live_preview} onChange=${(on) => saveSettings({ transcription: { live_preview: on } })} />
        <//>
      </div>
    <//>
  </div>`;
}

// Typing ---------------------------------------------------------------------------------------

const METHOD_OPTIONS = [
  { value: "paste", label: "Paste" },
  { value: "paste_shift_insert", label: "Paste with Shift + Insert" },
  { value: "type", label: "Type" },
];

function TypingSection() {
  const insertion = settings.value.insertion;
  const text = settings.value.text;
  return html`<div class="stack">
    <${Card}>
      <div class="rows">
        <${Row}
          label="How text goes in"
          hint=${insertion.method === "paste"
            ? "Paste is instant and works almost everywhere."
            : "Typing is slower, but works in apps that block pasting."}
        >
          <${Segmented}
            label="How text goes in"
            value=${insertion.method}
            options=${[
              { value: "paste", label: "Paste" },
              { value: "type", label: "Type" },
            ]}
            onChange=${(method) => saveSettings({ insertion: { method } })}
          />
        <//>
        <${Row} label="Restore the clipboard" hint="Put back whatever you'd copied before Whisp pasted.">
          <${Toggle}
            label="Restore the clipboard"
            checked=${insertion.restore_clipboard}
            disabled=${insertion.method === "type"}
            onChange=${(on) => saveSettings({ insertion: { restore_clipboard: on } })}
          />
        <//>
        <${Row} label="If you switch windows" hint="What happens when your words are ready but you've moved to another window.">
          <${Select}
            label="If you switch windows"
            value=${insertion.on_focus_change}
            options=${[
              { value: "clipboard", label: "Copy them to the clipboard" },
              { value: "insert", label: "Type them into the new window" },
            ]}
            onChange=${(value) => saveSettings({ insertion: { on_focus_change: value } })}
          />
        <//>
        <${Row} label="Space between dictations" hint="Auto adds a space when you carry on a sentence.">
          <${Segmented}
            label="Space between dictations"
            value=${text.smart_spacing}
            options=${[
              { value: "auto", label: "Auto" },
              { value: "always", label: "Always" },
              { value: "never", label: "Never" },
            ]}
            onChange=${(value) => saveSettings({ text: { smart_spacing: value } })}
          />
        <//>
      </div>
    <//>
    <${Card} title="Per-app overrides" description="Some apps, like terminals, need a different way to insert text.">
      <div class="card-body">
        <${AppOverrides}
          idPrefix="methods"
          value=${insertion.app_methods}
          options=${METHOD_OPTIONS}
          onChange=${(next) => saveSettings({ insertion: { app_methods: next } })}
        />
      </div>
    <//>
  </div>`;
}

// Overlay ----------------------------------------------------------------------------------------

function OverlayPreview({ position, idle }) {
  return html`<div class="screen-preview" aria-hidden="true">
    <div class="screen-preview-window"></div>
    <div class=${`screen-preview-pill is-${position}${idle ? "" : " is-hidden"}`}></div>
    <div class="screen-preview-taskbar"></div>
  </div>`;
}

function OverlaySection() {
  const overlay = settings.value.overlay;
  const set = (patch) => saveSettings({ overlay: patch });
  return html`<div class="stack">
    <${Card}>
      <div class="overlay-intro">
        <${OverlayPreview} position=${overlay.position} idle=${overlay.idle_pill} />
        <p class="overlay-intro-text">
          The overlay is the small pill that shows Whisp is listening. It never takes focus away from what you're typing in.
        </p>
      </div>
      <div class="rows">
        <${Row} label="Resting pill" hint="Show a tiny pill when Whisp is ready. Hover it to see your shortcut, or click it to start hands-free.">
          <${Toggle} label="Resting pill" checked=${overlay.idle_pill} onChange=${(on) => set({ idle_pill: on })} />
        <//>
        <${Row} label="Position">
          <${Segmented}
            label="Overlay position"
            value=${overlay.position}
            options=${[
              { value: "bottom", label: "Bottom", icon: "panel-bottom" },
              { value: "top", label: "Top", icon: "panel-top" },
            ]}
            onChange=${(position) => set({ position })}
          />
        <//>
        <${Row} label="Show on" hint="The screen you're typing on, or always your main screen.">
          <${Segmented}
            label="Show the overlay on"
            value=${overlay.show_on}
            options=${[
              { value: "active", label: "Active screen" },
              { value: "primary", label: "Main screen" },
            ]}
            onChange=${(show_on) => set({ show_on })}
          />
        <//>
        <${Row} label="Hide in fullscreen apps" hint="Keeps the pill out of games, videos and presentations.">
          <${Toggle} label="Hide in fullscreen apps" checked=${overlay.hide_in_fullscreen} onChange=${(on) => set({ hide_in_fullscreen: on })} />
        <//>
      </div>
    <//>
  </div>`;
}

// Privacy ----------------------------------------------------------------------------------------

const RETENTION = [
  { value: 1, label: "1 day" },
  { value: 7, label: "1 week" },
  { value: 30, label: "30 days" },
  { value: 90, label: "90 days" },
  { value: 365, label: "1 year" },
  { value: 0, label: "Forever" },
];

function PrivacySection() {
  const privacy = settings.value.privacy;
  const retention = RETENTION.some((item) => item.value === privacy.history_retention_days)
    ? RETENTION
    : [...RETENTION, { value: privacy.history_retention_days, label: plural(privacy.history_retention_days, "day") }];

  const clearNow = async () => {
    try {
      await call("history.clear");
      toast("History cleared.");
    } catch (error) {
      reportError(error);
    }
  };

  const clear = async () => {
    const ok = await confirmDialog({
      title: "Clear all history?",
      text: "This permanently deletes every saved dictation on this PC.",
      confirmLabel: "Clear history",
      danger: true,
    });
    if (ok) await clearNow();
  };

  const setHistoryEnabled = async (on) => {
    await saveSettings({ privacy: { history_enabled: on } });
    if (on) return;
    // Turning history off stops new saves; offer to delete what's already there.
    let saved = [];
    try {
      saved = (await call("history.list", { limit: 1 })).entries;
    } catch (error) {
      reportError(error);
      return;
    }
    if (!saved.length) return;
    const remove = await confirmDialog({
      title: "Delete the dictations already saved?",
      text: "History is off, so Whisp won't save anything new. What you saved before is still on this PC.",
      confirmLabel: "Delete them",
      cancelLabel: "Keep them",
      danger: true,
    });
    if (remove) await clearNow();
  };

  return html`<div class="stack">
    <div class="privacy-hero">
      <div class="privacy-hero-icon"><${Icon} name="shield-check" size=${20} /></div>
      <div>
        <h2 class="privacy-hero-title">Your voice stays on this PC</h2>
        <p class="privacy-hero-text">
          Whisp turns speech into text right here, without the internet. Nothing leaves your computer unless you connect an
          outside AI server for polish.
        </p>
      </div>
    </div>
    <${Card}>
      <div class="rows">
        <${Row} label="Save history" hint="Keep what you dictate so you can find and copy it later. Turned off, Whisp keeps only word counts, timings and app names for your stats.">
          <${Toggle} label="Save history" checked=${privacy.history_enabled} onChange=${setHistoryEnabled} />
        <//>
        <${Row} label="Keep history for" hint="Older dictations are deleted automatically.">
          <${Select}
            label="Keep history for"
            value=${privacy.history_retention_days}
            options=${retention}
            disabled=${!privacy.history_enabled}
            onChange=${(value) => saveSettings({ privacy: { history_retention_days: Number(value) } })}
          />
        <//>
        <${Row} label="Clear history" hint="Delete every saved dictation now.">
          <${Button} kind="danger-ghost" icon="trash-2" onClick=${clear}>Clear<//>
        <//>
        <${Row} label="Your data" hint="History, models and logs live in one folder.">
          <${Button} kind="secondary" icon="folder-open" onClick=${() => openFolder("data")}>Open folder<//>
        <//>
      </div>
    <//>
  </div>`;
}

// About -------------------------------------------------------------------------------------------

function Licence({ item }) {
  return html`<li class="licence">
    <div class="licence-line">
      <span class="licence-name">${item.name}${item.version ? html`<span class="faint"> ${item.version}</span>` : ""}</span>
      <span class="licence-meta">
        <${Badge}>${item.license}<//>
        ${item.url && html`<${IconButton} icon="arrow-up-right" label="Open website" tipSide="left" onClick=${() => openUrl(item.url)} />`}
      </span>
    </div>
    ${item.attribution && html`<p class="licence-attribution">${item.attribution}</p>`}
    ${item.text &&
    html`<details class="licence-text">
      <summary>Show licence</summary>
      <pre class="selectable">${item.text}</pre>
    </details>`}
  </li>`;
}

function AboutSection() {
  const info = appInfo.value;
  const licences = useResource(() => call("app.licenses"), []);
  return html`<div class="stack">
    <${Card}>
      <div class="about-hero">
        <img src="/brand/logo.svg" alt="" width="56" height="56" />
        <div>
          <h2 class="about-name">Whisp</h2>
          <p class="about-version tnum">Version ${info.version}</p>
          <p class="about-tag">Free and open source, under the MIT licence.</p>
        </div>
      </div>
      <div class="rows">
        <${Row} label="Logs" hint="Helpful when reporting a problem. They never include what you said.">
          <${Button} kind="secondary" icon="folder-open" onClick=${() => openFolder("logs")}>Open logs<//>
        <//>
        <${Row} label="Settings file" hint="Your preferences, stored with your Windows profile.">
          <${Button} kind="secondary" icon="folder-open" onClick=${() => openFolder("config")}>Open folder<//>
        <//>
      </div>
    <//>
    ${licences.error && html`<${Notice} kind="danger" title="Couldn't load the licences">${licences.error.message}<//>`}
    ${licences.data &&
    html`
      <${Card} title="Speech models" description="Models you can download, and the licences they're shared under.">
        <ul class="licences">${licences.data.models.map((item) => html`<${Licence} item=${item} />`)}</ul>
      <//>
      <${Card} title="AI polish" description="Downloaded only if you turn on built-in AI polish.">
        <ul class="licences">${(licences.data.ai_models || []).map((item) => html`<${Licence} item=${item} />`)}</ul>
      <//>
      <${Card} title="Open-source software" description="Whisp is built on these projects. Thank you to everyone behind them.">
        <ul class="licences">${licences.data.components.map((item) => html`<${Licence} item=${item} />`)}</ul>
      <//>
    `}
    <p class="about-build tnum">Qt ${info.qt} · Chromium ${info.chromium} · Python ${info.python} · Windows ${info.windows}</p>
  </div>`;
}

// Page --------------------------------------------------------------------------------------------

const VIEWS = {
  general: GeneralSection,
  shortcuts: ShortcutsSection,
  microphone: MicrophoneSection,
  model: ModelSection,
  typing: TypingSection,
  overlay: OverlaySection,
  privacy: PrivacySection,
  about: AboutSection,
};

export function SettingsPage() {
  const requested = route.value.section;
  const active = VIEWS[requested] ? requested : "general";
  const View = VIEWS[active];
  const tabs = useRef(null);
  const onKeyDown = (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const index = SECTIONS.findIndex((section) => section.id === active);
    const next = SECTIONS[(index + step + SECTIONS.length) % SECTIONS.length];
    navigate("settings", next.id);
    requestAnimationFrame(() => tabs.current?.querySelector('[aria-selected="true"]')?.focus());
  };
  return html`<div class="page settings">
    <header class="page-header">
      <div>
        <h1 class="page-title">Settings</h1>
        <p class="page-desc">Changes save as you make them.</p>
      </div>
    </header>
    <div class="tabs" role="tablist" aria-label="Settings sections" ref=${tabs} onKeyDown=${onKeyDown}>
      ${SECTIONS.map(
        (section) => html`<button
          class="tab"
          role="tab"
          id=${`tab-${section.id}`}
          aria-selected=${section.id === active ? "true" : "false"}
          aria-controls="settings-panel"
          tabindex=${section.id === active ? 0 : -1}
          onClick=${() => navigate("settings", section.id)}
        >
          ${section.label}
        </button>`,
      )}
    </div>
    <div class="settings-panel" id="settings-panel" role="tabpanel" aria-labelledby=${`tab-${active}`} key=${active}>
      <${View} />
    </div>
  </div>`;
}
