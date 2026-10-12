import { useEffect, useRef, useState } from "preact/hooks";
import { html } from "../html.js";
import { call } from "../bridge.js";
import { useEvent } from "../hooks.js";
import { Icon } from "../icons.js";
import { DeviceSelect, useDevices, useMicMeter } from "../mic.js";
import { GpuCard, ModelCard, catalog, catalogError, downloads, selectModel, useModels } from "../models.js";
import { ShortcutRecorder } from "../shortcut.js";
import { engine, navigate, reportError, route, saveSettings, settings } from "../store.js";
import { Button, Keycaps, Spinner } from "../ui.js";
import { Wisp } from "../wisp.js";

const STEPS = ["welcome", "microphone", "model", "shortcut", "practice", "done"];

function Welcome() {
  return html`<div class="ob-welcome">
    <div class="ob-wisp-hero"><${Wisp} demo height=${132} /></div>
    <h1 class="ob-title" tabindex="-1">Welcome to Whisp</h1>
    <p class="ob-text">Hold a shortcut, say what you mean, and clean text appears wherever you're typing.</p>
    <ul class="ob-points">
      <li><${Icon} name="shield-check" size=${18} /><span><strong>Private</strong>Speech becomes text on this PC.</span></li>
      <li><${Icon} name="zap" size=${18} /><span><strong>Fast</strong>Your words land the moment you stop.</span></li>
      <li><${Icon} name="app-window" size=${18} /><span><strong>Everywhere</strong>Works in any app you can type in.</span></li>
    </ul>
  </div>`;
}

function Microphone() {
  const audio = settings.value.audio;
  const devices = useDevices();
  const meter = useMicMeter(audio.input_device);
  const [peak, setPeak] = useState(0);
  const [quiet, setQuiet] = useState(false);

  useEffect(() => {
    setPeak(0);
    setQuiet(false);
    if (meter.state !== "started") return undefined;
    const timer = setTimeout(() => setQuiet(true), 6000);
    return () => clearTimeout(timer);
  }, [audio.input_device, meter.state]);
  useEvent("audio.level", (payload) => setPeak((value) => Math.max(value, payload.level)));

  let status;
  if (meter.state === "error") {
    status = html`<p class="ob-status is-error" role="alert">
      ${meter.message}
      <button class="link-btn" onClick=${meter.retry}>Try again</button>
      <button class="link-btn" onClick=${() => call("app.open_system_settings", { page: "microphone" }).catch(reportError)}>
        Check microphone access
      </button>
    </p>`;
  } else if (meter.state !== "started") {
    status = html`<p class="ob-status"><${Spinner} size=${14} /> Opening the microphone…</p>`;
  } else if (peak > 0.12) {
    status = html`<p class="ob-status is-good"><${Icon} name="circle-check" size=${15} /> Sounds good. Whisp can hear you.</p>`;
  } else if (quiet) {
    status = html`<p class="ob-status">Not moving? Check the microphone isn't muted, or pick another one.</p>`;
  } else {
    status = html`<p class="ob-status">Say a few words. The line moves as you speak.</p>`;
  }

  return html`<div class="ob-section">
    <h1 class="ob-title" tabindex="-1">Choose your microphone</h1>
    <p class="ob-text">Whisp listens only while you hold your shortcut.</p>
    <div class="ob-field">
      <${DeviceSelect}
        devices=${devices.data?.devices}
        value=${audio.input_device}
        onChange=${(device) => saveSettings({ audio: { input_device: device } })}
      />
    </div>
    <div class="ob-meter"><${Wisp} level=${meter.level} height=${112} label="Microphone level" /></div>
    ${status}
  </div>`;
}

function SpeechModel() {
  const data = useModels();
  const [showAll, setShowAll] = useState(false);

  // During setup, use whichever model finishes first, preferring the recommended one.
  useEffect(() => {
    if (!data) return;
    const current = data.models.find((model) => model.id === data.selected);
    if (current && current.installed) return;
    const recommended = data.models.find((model) => model.id === data.recommended?.id);
    const pick = recommended && recommended.installed ? recommended : data.models.find((model) => model.installed);
    if (pick) selectModel(pick);
  }, [data]);

  if (!data) {
    if (catalogError.value) return html`<p class="ob-status is-error">${catalogError.value.message}</p>`;
    return html`<p class="ob-status"><${Spinner} size=${14} /> Checking this PC…</p>`;
  }
  const recommended = data.models.find((model) => model.id === data.recommended?.id) || data.models[0];
  const others = data.models.filter((model) => model !== recommended);
  return html`<div class="ob-section">
    <h1 class="ob-title" tabindex="-1">Download a speech model</h1>
    <p class="ob-text">Whisp turns speech into text with a model that runs on your PC. Download one once and it works offline.</p>
    <${GpuCard} gpu=${data.gpu} />
    ${recommended && html`<${ModelCard} model=${recommended} data=${data} primary />`}
    ${others.length > 0 &&
    html`<button class="ob-more" aria-expanded=${showAll ? "true" : "false"} onClick=${() => setShowAll(!showAll)}>
      <${Icon} name=${showAll ? "chevron-down" : "chevron-right"} size=${14} />${showAll ? "Hide other models" : `See ${others.length} other ${others.length === 1 ? "model" : "models"}`}
    </button>`}
    ${showAll && html`<div class="models">${others.map((model) => html`<${ModelCard} key=${model.id} model=${model} data=${data} />`)}</div>`}
  </div>`;
}

function Shortcut() {
  const chord = settings.value.hotkeys.dictate;
  return html`<div class="ob-section">
    <h1 class="ob-title" tabindex="-1">Your dictation shortcut</h1>
    <p class="ob-text">Use it in any app, whenever you'd rather talk than type.</p>
    <div class="ob-shortcut">
      <${ShortcutRecorder}
        action="dictate"
        chord=${chord}
        size="lg"
        onSave=${(next) => saveSettings({ hotkeys: { dictate: next } })}
      />
    </div>
    <div class="ob-modes">
      <div class="ob-mode">
        <div class="ob-mode-icon"><${Icon} name="hand" size=${18} /></div>
        <h2>Hold to talk</h2>
        <p>Hold the keys while you speak. Let go, and your words appear.</p>
      </div>
      <div class="ob-mode">
        <div class="ob-mode-icon"><${Icon} name="mouse-pointer-click" size=${18} /></div>
        <h2>Tap for hands-free</h2>
        <p>Tap once and keep talking. Tap again when you're done. Esc cancels.</p>
      </div>
    </div>
  </div>`;
}

function engineLine(status, progress) {
  if (!status || status.state === "not_installed") {
    if (progress) {
      const pct = progress.total ? Math.floor((progress.done / progress.total) * 100) : 0;
      return html`<${Spinner} size=${14} /> Your speech model is downloading (${pct}%). You can try this once it's done.`;
    }
    return "Download a speech model first, then come back to try it.";
  }
  if (status.state === "loading") return html`<${Spinner} size=${14} /> Getting the speech model ready…`;
  if (status.state === "error") return status.message || "The speech engine has a problem. Check Settings → Speech model.";
  return null;
}

function Practice() {
  const chord = settings.value.hotkeys.dictate;
  const [done, setDone] = useState(false);
  const box = useRef(null);
  useEvent("history.added", () => setDone(true));
  useEffect(() => {
    box.current?.focus();
  }, []);
  const status = engine.value;
  const progress = Object.values(downloads.value)[0];
  const blocker = engineLine(status, progress);
  return html`<div class="ob-section">
    <h1 class="ob-title" tabindex="-1">Give it a try</h1>
    <p class="ob-text">
      Click in the box, hold <${Keycaps} chord=${chord} /> and say something like “Hi, this is my first dictation with Whisp.”
    </p>
    <textarea
      ref=${box}
      class="textarea ob-practice"
      rows="4"
      placeholder="Your words will appear here."
      aria-label="Practice dictation"
      onInput=${(event) => event.currentTarget.value.trim() && setDone(true)}
    ></textarea>
    ${done
      ? html`<p class="ob-status is-good"><${Icon} name="circle-check" size=${15} /> That's it. You're dictating.</p>`
      : blocker && html`<p class="ob-status">${blocker}</p>`}
  </div>`;
}

function Done() {
  const hotkeys = settings.value.hotkeys;
  return html`<div class="ob-done">
    <div class="ob-check" aria-hidden="true">
      <svg viewBox="0 0 52 52" width="52" height="52"><path d="M14 27l8 8 16-17" /></svg>
    </div>
    <h1 class="ob-title" tabindex="-1">You're all set</h1>
    <p class="ob-text">Whisp keeps running in the system tray, ready whenever you are.</p>
    <ul class="ob-tips">
      <li><${Keycaps} chord=${hotkeys.dictate} /><span>Hold to dictate, tap for hands-free</span></li>
      ${hotkeys.hands_free && html`<li><${Keycaps} chord=${hotkeys.hands_free} /><span>Start or stop hands-free dictation</span></li>`}
      ${hotkeys.paste_last && html`<li><${Keycaps} chord=${hotkeys.paste_last} /><span>Paste your last dictation again</span></li>`}
      <li><${Keycaps} keys=${["Esc"]} /><span>Cancel while Whisp is listening</span></li>
    </ul>
  </div>`;
}

export function OnboardingPage() {
  const requested = route.value.section;
  const [step, setStep] = useState(STEPS.includes(requested) ? requested : "welcome");
  const index = STEPS.indexOf(step);
  const finishedBefore = settings.value.general.onboarding_completed;
  const stage = useRef(null);

  useEffect(() => {
    if (STEPS.includes(requested)) setStep(requested);
  }, [requested]);
  useEffect(() => {
    stage.current?.scrollTo(0, 0);
    stage.current?.querySelector(".ob-title")?.focus({ preventScroll: true });
  }, [step]);

  const finish = async () => {
    try {
      if (!finishedBefore) {
        const next = await call("onboarding.complete");
        settings.value = next;
      }
      navigate("home");
    } catch (error) {
      reportError(error);
    }
  };

  const status = engine.value;
  const modelReady =
    Boolean(catalog.value?.models.some((model) => model.installed)) || Boolean(status && status.state !== "not_installed");
  const downloading = Object.keys(downloads.value).length > 0;

  let next = null;
  if (step === "welcome") next = html`<${Button} kind="primary" size="lg" iconAfter="arrow-right" onClick=${() => setStep("microphone")}>Get started<//>`;
  else if (step === "done") next = html`<${Button} kind="primary" size="lg" onClick=${finish}>Start using Whisp<//>`;
  else {
    const blocked = step === "model" && !modelReady && !downloading;
    next = html`<${Button}
      kind="primary"
      size="lg"
      iconAfter="arrow-right"
      disabled=${blocked}
      onClick=${() => setStep(STEPS[index + 1])}
    >
      Continue
    <//>`;
  }

  return html`<div class="onboarding">
    <header class="ob-top">
      <div class="brand"><img src="/brand/logo.svg" alt="" /><span class="brand-name">Whisp</span></div>
      <ol class="ob-progress" aria-label=${`Step ${index + 1} of ${STEPS.length}`}>
        ${STEPS.map((name, i) => html`<li class=${`ob-pip${i < index ? " is-done" : ""}${i === index ? " is-current" : ""}`}></li>`)}
      </ol>
      <div class="ob-top-end">
        ${step !== "done" &&
        html`<${Button} kind="ghost" size="sm" onClick=${finish}>${finishedBefore ? "Close" : "Skip setup"}<//>`}
      </div>
    </header>
    <div class="ob-stage" ref=${stage}>
      <div class="ob-step" key=${step}>
        ${step === "welcome" && html`<${Welcome} />`}
        ${step === "microphone" && html`<${Microphone} />`}
        ${step === "model" && html`<${SpeechModel} />`}
        ${step === "shortcut" && html`<${Shortcut} />`}
        ${step === "practice" && html`<${Practice} />`}
        ${step === "done" && html`<${Done} />`}
      </div>
    </div>
    <footer class="ob-footer">
      <div>
        ${index > 0 &&
        step !== "done" &&
        html`<${Button} kind="ghost" icon="arrow-left" onClick=${() => setStep(STEPS[index - 1])}>Back<//>`}
      </div>
      ${step === "model" && !modelReady && !downloading && html`<span class="ob-footer-note">Download a model to continue</span>`}
      ${next}
    </footer>
  </div>`;
}
