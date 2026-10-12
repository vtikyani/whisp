// Shared building blocks. Every control is keyboard accessible and labelled.
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { signal } from "@preact/signals";
import { html } from "./html.js";
import { Icon } from "./icons.js";
import { dismissToast, keysFor, toasts } from "./store.js";

const cx = (...names) => names.filter(Boolean).join(" ");

export function Button({ kind = "secondary", size, icon, iconAfter, children, class: className, busy, ...rest }) {
  return html`<button
    type="button"
    class=${cx("btn", `btn-${kind}`, size && `btn-${size}`, className)}
    aria-busy=${busy ? "true" : undefined}
    ...${rest}
  >
    ${busy ? html`<${Icon} name="loader-circle" class="spin" />` : icon && html`<${Icon} name=${icon} />`}
    ${children}
    ${iconAfter && html`<${Icon} name=${iconAfter} />`}
  </button>`;
}

export function IconButton({ icon, label, danger, tipSide, class: className, size = 16, ...rest }) {
  return html`<button
    type="button"
    class=${cx("icon-btn", danger && "is-danger", className)}
    aria-label=${label}
    data-tip=${label}
    data-tip-side=${tipSide}
    ...${rest}
  >
    <${Icon} name=${icon} size=${size} />
  </button>`;
}

export function Toggle({ checked, onChange, label, disabled }) {
  return html`<button
    type="button"
    class="toggle"
    role="switch"
    aria-checked=${checked ? "true" : "false"}
    aria-label=${label}
    disabled=${disabled}
    onClick=${() => onChange(!checked)}
  ></button>`;
}

export function Segmented({ value, options, onChange, label, size, disabled }) {
  const root = useRef(null);
  const [box, setBox] = useState(null);
  const settled = useRef(false);

  useLayoutEffect(() => {
    const element = root.current;
    if (!element) return undefined;
    const measure = () => {
      const active = element.querySelector('[aria-checked="true"]');
      setBox(active ? { x: active.offsetLeft, width: active.offsetWidth } : null);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [value, options.map((o) => o.label).join("|")]);

  useEffect(() => {
    if (box) settled.current = true;
  }, [box]);

  const enabled = options.filter((o) => !o.disabled && !disabled);
  const onKeyDown = (event) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!step || !enabled.length) return;
    event.preventDefault();
    const index = enabled.findIndex((o) => o.value === value);
    const next = enabled[(index + step + enabled.length) % enabled.length];
    onChange(next.value);
    requestAnimationFrame(() => root.current?.querySelector('[aria-checked="true"]')?.focus());
  };

  return html`<div
    class=${cx("segmented", size && `segmented-${size}`)}
    role="radiogroup"
    aria-label=${label}
    ref=${root}
    onKeyDown=${onKeyDown}
  >
    ${box &&
    html`<span
      class=${cx("segmented-indicator", !settled.current && "is-instant")}
      style=${{ width: `${box.width}px`, transform: `translateX(${box.x}px)` }}
    ></span>`}
    ${options.map(
      (option) => html`<button
        type="button"
        role="radio"
        aria-checked=${option.value === value ? "true" : "false"}
        tabindex=${option.value === value ? 0 : -1}
        disabled=${disabled || option.disabled}
        title=${option.title}
        onClick=${() => option.value !== value && onChange(option.value)}
      >
        ${option.icon && html`<${Icon} name=${option.icon} size=${14} />`}${option.label}
      </button>`,
    )}
  </div>`;
}

export function Select({ value, options, onChange, label, disabled, size, class: className, style }) {
  return html`<span class=${cx("select", size && `select-${size}`, className)} style=${style}>
    <select
      aria-label=${label}
      disabled=${disabled}
      onChange=${(event) => onChange(event.currentTarget.value)}
    >
      ${options.map(
        (option) =>
          html`<option value=${option.value} selected=${String(option.value) === String(value)} disabled=${option.disabled}>
            ${option.label}
          </option>`,
      )}
    </select>
    <${Icon} name="chevron-down" size=${14} />
  </span>`;
}

/** A range slider that saves when released, showing the live value while dragging. */
export function Slider({ value, min, max, step = 1, onCommit, label, format = String, width }) {
  const [draft, setDraft] = useState(value);
  const dragging = useRef(false);
  useEffect(() => {
    if (!dragging.current) setDraft(value);
  }, [value]);
  const pct = ((draft - min) / (max - min)) * 100;
  const commit = (next) => {
    dragging.current = false;
    if (next !== value) onCommit(next);
  };
  return html`<div class="slider" style=${width ? { width } : undefined}>
    <input
      type="range"
      class="range"
      min=${min}
      max=${max}
      step=${step}
      value=${draft}
      aria-label=${label}
      aria-valuetext=${format(draft)}
      style=${{ "--pct": `${pct}%` }}
      onInput=${(event) => {
        dragging.current = true;
        setDraft(Number(event.currentTarget.value));
      }}
      onChange=${(event) => commit(Number(event.currentTarget.value))}
    />
    <span class="slider-value">${format(draft)}</span>
  </div>`;
}

/** A text input that saves on blur or Enter, and reverts on Esc. */
export function TextField({ value, onCommit, label, placeholder, type = "text", maxLength, class: className, icon, autoFocus, invalid }) {
  const [draft, setDraft] = useState(value ?? "");
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setDraft(value ?? "");
  }, [value]);
  const commit = () => {
    focused.current = false;
    const next = draft.trim();
    if (next !== (value ?? "")) onCommit(next);
    else setDraft(value ?? "");
  };
  const input = html`<input
    class=${cx("input", invalid && "is-invalid", !icon && className)}
    type=${type}
    value=${draft}
    placeholder=${placeholder}
    aria-label=${label}
    maxlength=${maxLength}
    autofocus=${autoFocus}
    spellcheck="false"
    onFocus=${() => {
      focused.current = true;
    }}
    onInput=${(event) => setDraft(event.currentTarget.value)}
    onBlur=${commit}
    onKeyDown=${(event) => {
      if (event.key === "Enter") event.currentTarget.blur();
      if (event.key === "Escape") {
        setDraft(value ?? "");
        focused.current = false;
        event.currentTarget.blur();
      }
    }}
  />`;
  return icon ? html`<span class=${cx("input-icon", className)}><${Icon} name=${icon} />${input}</span>` : input;
}

export function Row({ label, hint, children, top, class: className }) {
  return html`<div class=${cx("row", top && "is-top", className)}>
    <div class="row-text">
      <div class="row-label">${label}</div>
      ${hint && html`<div class="row-hint">${hint}</div>`}
    </div>
    ${children && html`<div class="row-control">${children}</div>`}
  </div>`;
}

export function Card({ title, description, actions, children, class: className, icon }) {
  return html`<section class=${cx("card", className)}>
    ${(title || actions) &&
    html`<header class="card-header">
      <div>
        ${title && html`<h2 class="card-title">${icon && html`<${Icon} name=${icon} size=${16} />`}${title}</h2>`}
        ${description && html`<p class="card-desc">${description}</p>`}
      </div>
      ${actions && html`<div class="row-control">${actions}</div>`}
    </header>`}
    ${children}
  </section>`;
}

export function Keycaps({ chord, keys, size }) {
  const labels = keys || keysFor(chord);
  if (!labels.length) return html`<span class="keys-empty">Not set</span>`;
  return html`<span class=${cx("keys", size === "lg" && "keys-lg")} role="img" aria-label=${labels.join(" + ")}>
    ${labels.map((label) => html`<kbd class="key">${label}</kbd>`)}
  </span>`;
}

export function Badge({ kind, icon, children }) {
  return html`<span class=${cx("badge", kind && `badge-${kind}`)}>
    ${icon && html`<${Icon} name=${icon} size=${12} />`}${children}
  </span>`;
}

export function Progress({ done = 0, total = 0, indeterminate, label }) {
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  return html`<div
    class=${cx("progress", indeterminate && "is-indeterminate")}
    role="progressbar"
    aria-label=${label}
    aria-valuemin="0"
    aria-valuemax="100"
    aria-valuenow=${indeterminate ? undefined : Math.round(pct)}
  >
    <div class="progress-fill" style=${{ "--pct": `${pct}%` }}></div>
  </div>`;
}

export function Pips({ value, max = 5, label }) {
  return html`<span class="pips" role="img" aria-label=${`${label}: ${value} out of ${max}`}>
    ${Array.from({ length: max }, (_, i) => html`<span class=${cx("pip", i < value && "is-on")}></span>`)}
  </span>`;
}

export function Spinner({ size = 16 }) {
  return html`<${Icon} name="loader-circle" size=${size} class="spin" />`;
}

export function Notice({ kind, icon, title, children, actions }) {
  const glyph = icon || (kind === "danger" ? "circle-alert" : kind === "warning" ? "triangle-alert" : "info");
  return html`<div class=${cx("notice", kind && `notice-${kind}`)} role=${kind === "danger" ? "alert" : undefined}>
    <${Icon} name=${glyph} />
    <div class="notice-body">
      ${title && html`<div class="notice-title">${title}</div>`}
      ${children && html`<div>${children}</div>`}
    </div>
    ${actions && html`<div class="notice-actions">${actions}</div>`}
  </div>`;
}

export function Empty({ icon, title, children, action }) {
  return html`<div class="empty">
    <div class="empty-icon"><${Icon} name=${icon} /></div>
    <h3 class="empty-title">${title}</h3>
    ${children && html`<p class="empty-text">${children}</p>`}
    ${action}
  </div>`;
}

export function FieldError({ children }) {
  if (!children) return null;
  return html`<div class="field-error" role="alert"><${Icon} name="circle-alert" />${children}</div>`;
}

// Toasts ------------------------------------------------------------------------

const TOAST_ICONS = { info: "info", success: "circle-check", error: "circle-alert" };

function ToastItem({ item }) {
  const [leaving, setLeaving] = useState(false);
  const remaining = useRef(item.duration);
  const startedAt = useRef(0);
  const timer = useRef(null);
  const close = () => {
    clearTimeout(timer.current);
    setLeaving(true);
    setTimeout(() => dismissToast(item.id), 160);
  };
  const start = () => {
    startedAt.current = Date.now();
    clearTimeout(timer.current);
    timer.current = setTimeout(close, remaining.current);
  };
  const pause = () => {
    clearTimeout(timer.current);
    remaining.current = Math.max(1200, remaining.current - (Date.now() - startedAt.current));
  };
  useEffect(() => {
    start();
    return () => clearTimeout(timer.current);
  }, []);
  return html`<div
    class=${cx("toast", `toast-${item.kind}`, leaving && "is-leaving")}
    role=${item.kind === "error" ? "alert" : "status"}
    onMouseEnter=${pause}
    onMouseLeave=${start}
  >
    <${Icon} name=${TOAST_ICONS[item.kind] || "info"} />
    <div class="toast-message">${item.message}</div>
    ${item.action &&
    html`<${Button}
      kind="ghost"
      size="sm"
      onClick=${() => {
        item.action.run();
        close();
      }}
      >${item.action.label}<//
    >`}
    <${IconButton} icon="x" label="Dismiss" size=${14} onClick=${close} />
  </div>`;
}

export function Toaster() {
  return html`<div class="toaster" aria-live="polite">
    ${toasts.value.map((item) => html`<${ToastItem} key=${item.id} item=${item} />`)}
  </div>`;
}

// Confirm dialog -------------------------------------------------------------------

const dialogState = signal(null);

/** Ask a yes/no question in a modal. Resolves true when confirmed. */
export function confirmDialog({ title, text, confirmLabel, cancelLabel = "Cancel", danger = false }) {
  return new Promise((resolve) => {
    dialogState.value = { title, text, confirmLabel, cancelLabel, danger, resolve };
  });
}

export function DialogHost() {
  const state = dialogState.value;
  const ref = useRef(null);
  useEffect(() => {
    const dialog = ref.current;
    if (state && dialog && !dialog.open) dialog.showModal();
  }, [state]);
  if (!state) return null;
  const finish = (result) => {
    ref.current?.close();
    dialogState.value = null;
    state.resolve(result);
  };
  return html`<dialog
    class="dialog"
    ref=${ref}
    aria-labelledby="dialog-title"
    aria-describedby=${state.text ? "dialog-text" : undefined}
    onCancel=${(event) => {
      event.preventDefault();
      finish(false);
    }}
    onClick=${(event) => {
      if (event.target === event.currentTarget) finish(false);
    }}
  >
    <div class="dialog-body">
      <h2 class="dialog-title" id="dialog-title">${state.title}</h2>
      ${state.text && html`<p class="dialog-text" id="dialog-text">${state.text}</p>`}
    </div>
    <div class="dialog-actions">
      <${Button} kind="secondary" autofocus=${state.danger} onClick=${() => finish(false)}>${state.cancelLabel}<//>
      <${Button} kind=${state.danger ? "danger" : "primary"} autofocus=${!state.danger} onClick=${() => finish(true)}
        >${state.confirmLabel}<//
      >
    </div>
  </dialog>`;
}
