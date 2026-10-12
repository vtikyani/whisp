// Microphone: device list, and a live level meter that runs only while it's on screen.
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { signal } from "@preact/signals";
import { html } from "./html.js";
import { call } from "./bridge.js";
import { useEvent, useResource } from "./hooks.js";
import { Select } from "./ui.js";

/** Opens the microphone for a level meter while `active`; stops when hidden or unmounted. */
export function useMicMeter(device, active = true) {
  const level = useMemo(() => signal(0), []);
  const [status, setStatus] = useState({ state: "starting" });
  const [attempt, setAttempt] = useState(0);
  const running = useRef(false);

  useEvent("audio.level", (payload) => {
    if (running.current) level.value = payload.level;
  });
  useEvent("audio.meter", (payload) => {
    if (!running.current) return;
    setStatus(payload);
    if (payload.state !== "started") level.value = 0;
  });

  useEffect(() => {
    if (!active) return undefined;
    const start = () => {
      running.current = true;
      setStatus({ state: "starting" });
      call("audio.meter_start", { device: device || null }).catch((error) =>
        setStatus({ state: "error", message: error.message }),
      );
    };
    const stop = () => {
      running.current = false;
      level.value = 0;
      call("audio.meter_stop").catch(() => {});
    };
    const onVisibility = () => (document.hidden ? stop() : start());
    if (!document.hidden) start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      stop();
    };
  }, [device, active, attempt]);

  return { level, ...status, retry: () => setAttempt((n) => n + 1) };
}

export function useDevices() {
  return useResource(() => call("audio.devices"), []);
}

export function deviceOptions(devices, selected) {
  const list = devices || [];
  const fallback = list.find((device) => device.is_default);
  const options = [
    { value: "", label: fallback ? `Windows default · ${fallback.name}` : "Windows default" },
    ...list.map((device) => ({ value: device.id, label: device.name })),
  ];
  if (selected && !list.some((device) => device.id === selected)) {
    options.push({ value: selected, label: `${selected} (not connected)` });
  }
  return options;
}

export function DeviceSelect({ devices, value, onChange, disabled }) {
  return html`<${Select}
    label="Microphone"
    value=${value || ""}
    options=${deviceOptions(devices, value)}
    onChange=${(next) => onChange(next || null)}
    disabled=${disabled}
    class="device-select"
  />`;
}

/** A precise horizontal level meter (the wisp is the expressive version). */
export function LevelBar({ level, label = "Microphone level" }) {
  const fill = useRef(null);
  const smoothed = useRef(0);
  useEffect(() => {
    let frame = 0;
    let last = performance.now();
    const tick = (now) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const target = level.value;
      const tau = target > smoothed.current ? 0.05 : 0.22;
      smoothed.current += (target - smoothed.current) * (1 - Math.exp(-dt / tau));
      if (fill.current) fill.current.style.setProperty("--pct", `${Math.min(100, smoothed.current * 100)}%`);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [level]);
  return html`<div class="level" role="meter" aria-label=${label} aria-valuemin="0" aria-valuemax="100">
    <div class="level-fill" ref=${fill}></div>
  </div>`;
}
