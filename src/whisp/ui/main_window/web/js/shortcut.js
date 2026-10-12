// Shows a shortcut as key caps and records a new one through HotkeyService.
import { useEffect, useRef, useState } from "preact/hooks";
import { html } from "./html.js";
import { call } from "./bridge.js";
import { useEvent } from "./hooks.js";
import { ensureKeys } from "./store.js";
import { Button, FieldError, IconButton, Keycaps } from "./ui.js";

export function ShortcutRecorder({ action, chord, onSave, onClear, error: bindingError, size, changeLabel = "Change" }) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState(null);
  const recordingRef = useRef(false);
  recordingRef.current = recording;

  const stop = () => {
    setRecording(false);
    call("hotkeys.record_cancel").catch(() => {});
  };

  useEvent("hotkeys.recorded", async (payload) => {
    if (!recordingRef.current) return;
    setRecording(false);
    // HotkeyService sends "" when the user pressed Esc to cancel.
    if (!payload.chord || ["esc", "escape"].includes(payload.chord)) return;
    try {
      const result = await call("hotkeys.validate", { chord: payload.chord, action });
      if (!result.valid) {
        setError(result.message);
        return;
      }
      setError(null);
      await ensureKeys([result.chord]);
      if (result.chord !== chord) onSave(result.chord);
    } catch (failure) {
      setError(failure.message);
    }
  });

  useEffect(() => {
    if (!recording) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        stop();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [recording]);

  useEffect(
    () => () => {
      if (recordingRef.current) call("hotkeys.record_cancel").catch(() => {});
    },
    [],
  );

  const start = async () => {
    setError(null);
    try {
      await call("hotkeys.record_start");
      setRecording(true);
    } catch (failure) {
      setError(failure.message);
    }
  };

  const message = error || bindingError;
  return html`<div class="shortcut">
    <div class="shortcut-line">
      ${recording
        ? html`<span class=${`shortcut-listening${size === "lg" ? " is-lg" : ""}`} role="status">
            <span class="rec-dot"></span>Press the new shortcut… <span class="faint">Esc cancels</span>
          </span>`
        : html`<${Keycaps} chord=${chord} size=${size} />`}
      <div class="shortcut-actions">
        ${recording
          ? html`<${Button} kind="ghost" size="sm" onClick=${stop}>Cancel<//>`
          : html`<${Button} kind="secondary" size="sm" onClick=${start}>${changeLabel}<//>`}
        ${onClear && chord && !recording
          ? html`<${IconButton} icon="x" label="Turn off this shortcut" tipSide="left" onClick=${onClear} />`
          : size !== "lg" && html`<span class="shortcut-slot" aria-hidden="true"></span>`}
      </div>
    </div>
    ${message && html`<${FieldError}>${message}<//>`}
  </div>`;
}
