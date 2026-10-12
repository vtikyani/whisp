// App-wide state as signals, plus the actions that change it.
import { signal, batch } from "@preact/signals";
import { call, on } from "./bridge.js";

export const PAGES = ["home", "history", "dictionary", "snippets", "styles", "settings", "onboarding"];

export const settings = signal(null);
export const engine = signal(null);
/** {paused, problem} from AppController. */
export const appStatus = signal({ paused: false, problem: null });
export const appInfo = signal(null);
export const theme = signal(document.documentElement.dataset.theme || "dark");
export const reducedMotion = signal(document.documentElement.dataset.motion === "reduced");
export const route = signal({ page: "home", section: null });
export const keyLabels = signal(new Map());
export const toasts = signal([]);

const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
let windowsReducedMotion = reducedMotion.value;

export function navigate(page, section = null) {
  if (!PAGES.includes(page)) return;
  const current = route.value;
  if (current.page === page && current.section === section) return;
  route.value = { page, section };
}

function applyTheme(value) {
  const next = value === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  theme.value = next;
}

function applyMotion() {
  const reduced = windowsReducedMotion || motionQuery.matches;
  document.documentElement.dataset.motion = reduced ? "reduced" : "full";
  reducedMotion.value = reduced;
}

export function hotkeyChords(value = settings.value) {
  if (!value) return [];
  const { dictate, command, paste_last, copy_last, hands_free } = value.hotkeys;
  return [dictate, command, paste_last, copy_last, hands_free].filter(Boolean);
}

export async function boot() {
  const ready = await call("app.ready");
  await ensureKeys(hotkeyChords(ready.settings));
  windowsReducedMotion = Boolean(ready.reduced_motion);
  batch(() => {
    settings.value = ready.settings;
    engine.value = ready.engine;
    appStatus.value = ready.status;
    appInfo.value = ready.info;
    applyTheme(ready.theme);
    applyMotion();
    const page = ready.settings.general.onboarding_completed ? ready.page : "onboarding";
    route.value = { page, section: ready.section || null };
  });
  on("settings.changed", (payload) => {
    settings.value = payload.settings;
    ensureKeys(hotkeyChords(payload.settings));
  });
  on("theme.changed", (payload) => applyTheme(payload.theme));
  on("motion.changed", (payload) => {
    windowsReducedMotion = Boolean(payload.reduced);
    applyMotion();
  });
  on("navigate", (payload) => navigate(payload.page, payload.section || null));
  on("engine.status", (payload) => {
    engine.value = payload;
  });
  on("app.status", (payload) => {
    appStatus.value = payload;
  });
  motionQuery.addEventListener("change", applyMotion);
}

// Settings ---------------------------------------------------------------

// Mapping settings that a patch replaces outright (mirrors settings.py).
const REPLACE_WHOLE = new Set(["app_categories", "app_methods"]);

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function mergePatch(base, patch) {
  const out = { ...base };
  for (const [key, value] of Object.entries(patch)) {
    if (isPlainObject(value) && isPlainObject(base[key]) && !REPLACE_WHOLE.has(key)) {
      out[key] = mergePatch(base[key], value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

function withoutSecret(patch) {
  if (!patch.polish || !("openai_api_key" in patch.polish)) return patch;
  const { openai_api_key: _secret, ...polish } = patch.polish;
  return { ...patch, polish };
}

/** Save a partial settings change. The UI updates at once and rolls back if Python refuses. */
export async function saveSettings(patch) {
  if (settings.value) settings.value = mergePatch(settings.value, withoutSecret(patch));
  try {
    settings.value = await call("settings.update", patch);
    return true;
  } catch (error) {
    reportError(error);
    try {
      settings.value = await call("settings.get");
    } catch {
      // The toast above already told the user; the next settings.changed event resyncs.
    }
    return false;
  }
}

// Hotkey labels -------------------------------------------------------------

function fallbackKeys(chord) {
  return chord
    .split("+")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => (part.length <= 3 ? part.toUpperCase() : part[0].toUpperCase() + part.slice(1)));
}

/** Key-cap labels for chords, as Windows names them (from keys.display_chord). */
export async function ensureKeys(chords) {
  const missing = [...new Set(chords)].filter((chord) => chord && !keyLabels.value.has(chord));
  if (!missing.length) return;
  const labels = await Promise.all(
    missing.map((chord) =>
      call("hotkeys.display", { chord }).then(
        (result) => result.keys,
        () => fallbackKeys(chord),
      ),
    ),
  );
  const next = new Map(keyLabels.value);
  missing.forEach((chord, index) => next.set(chord, labels[index]));
  keyLabels.value = next;
}

export function keysFor(chord) {
  if (!chord) return [];
  return keyLabels.value.get(chord) || fallbackKeys(chord);
}

// Toasts ----------------------------------------------------------------------

let nextToastId = 1;

export function toast(message, { kind = "info", action = null, duration = null } = {}) {
  const id = nextToastId++;
  const lifetime = duration ?? (kind === "error" ? 7000 : action ? 6000 : 3500);
  toasts.value = [...toasts.value.slice(-2), { id, message, kind, action, duration: lifetime }];
  return id;
}

export function dismissToast(id) {
  toasts.value = toasts.value.filter((item) => item.id !== id);
}

export function reportError(error) {
  toast((error && error.message) || "Something went wrong.", { kind: "error" });
}
