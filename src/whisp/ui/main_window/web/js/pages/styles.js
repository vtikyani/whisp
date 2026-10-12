import { useEffect, useState } from "preact/hooks";
import { html } from "../html.js";
import { call } from "../bridge.js";
import { formatBytes } from "../format.js";
import { useEvent, useResource } from "../hooks.js";
import { Icon } from "../icons.js";
import { AppOverrides } from "../overrides.js";
import { reportError, saveSettings, settings, toast } from "../store.js";
import {
  Badge,
  Button,
  Card,
  FieldError,
  IconButton,
  Notice,
  Progress,
  Row,
  Segmented,
  Select,
  Slider,
  Spinner,
  TextField,
  Toggle,
  confirmDialog,
} from "../ui.js";

export const CATEGORIES = [
  { id: "messaging", label: "Messaging", icon: "message-square", hint: "Chat apps like Slack, Teams and WhatsApp" },
  { id: "email", label: "Email", icon: "mail", hint: "Outlook, Gmail and other mail apps" },
  { id: "code", label: "Code", icon: "code", hint: "Code editors and terminals" },
  { id: "document", label: "Documents", icon: "file-text", hint: "Word, Google Docs, notes and writing apps" },
  { id: "other", label: "Other apps", icon: "app-window", hint: "Anything Whisp doesn't recognise" },
];

const STYLES = [
  { value: "verbatim", label: "Verbatim", title: "Exactly what you said" },
  { value: "clean", label: "Clean", title: "Fillers removed, tidy punctuation" },
  { value: "casual", label: "Casual", title: "Relaxed, like a chat message" },
  { value: "formal", label: "Formal", title: "Full sentences and proper punctuation" },
];

function CategoryStyle({ category, style, version }) {
  const preview = useResource(() => call("styles.preview", { category: category.id }), [category.id, version]);
  const after = preview.data?.styles?.[style];
  return html`<div class="style-row">
    <div class="style-head">
      <div class="style-icon"><${Icon} name=${category.icon} size=${16} /></div>
      <div class="style-text">
        <div class="row-label">${category.label}</div>
        <div class="row-hint">${category.hint}</div>
      </div>
      <${Segmented}
        label=${`Style for ${category.label}`}
        value=${style}
        options=${STYLES}
        size="sm"
        onChange=${(next) => saveSettings({ text: { styles: { [category.id]: next } } })}
      />
    </div>
    ${preview.data &&
    html`<div class="style-preview" aria-live="polite">
      <p class="style-before"><span class="style-tag">You say</span>${preview.data.before}</p>
      <p class="style-after"><span class="style-tag">Whisp types</span><span class="selectable">${after}</span></p>
    </div>`}
  </div>`;
}

// AI polish ----------------------------------------------------------------------------------

function statusLine(status) {
  if (!status) return null;
  if (status.state === "ready") return status.backend ? `Running on your ${status.backend}` : "Running";
  if (status.state === "starting") return status.message || "Starting…";
  return status.message;
}

function BuiltinPanel({ polish }) {
  const catalog = useResource(() => call("llm.catalog"), []);
  const [progress, setProgress] = useState(null);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    if (catalog.data) setStatus(catalog.data.status);
  }, [catalog.data]);
  useEvent("llm.progress", (payload) => setProgress(payload));
  useEvent("llm.status", (payload) => {
    setStatus(payload);
    if (!payload.installing) setProgress(null);
    catalog.reload(); // installs and removals finish in the background, then report a new status
  });
  useEvent("llm.installed", (payload) => {
    setProgress(null);
    const model = catalog.data?.models.find((item) => item.id === payload.id);
    toast(`${model ? model.name : "The model"} is ready.`, { kind: "success" });
    if (!settings.value.polish.builtin_model) saveSettings({ polish: { builtin_model: payload.id } });
    catalog.reload();
  });
  useEvent("llm.failed", (payload) => {
    setProgress(null);
    toast(payload.message, { kind: payload.message === "Download cancelled." ? "info" : "error" });
    catalog.reload();
  });

  const installing = status?.installing || null;
  const install = async (model) => {
    setProgress({ id: model.id, done: 0, total: model.size || 0 });
    try {
      await call("llm.install", { id: model.id });
    } catch (error) {
      setProgress(null);
      reportError(error);
    }
  };
  const cancel = async () => {
    try {
      await call("llm.cancel");
    } catch (error) {
      reportError(error);
    }
  };
  const uninstall = async (model) => {
    const ok = await confirmDialog({
      title: `Remove ${model.name}?`,
      text: "This frees up disk space. You can download it again later.",
      confirmLabel: "Remove",
      danger: true,
    });
    if (!ok) return;
    try {
      await call("llm.uninstall", { id: model.id });
      if (polish.builtin_model === model.id) saveSettings({ polish: { builtin_model: "" } });
      toast(`${model.name} removed.`);
      catalog.reload();
    } catch (error) {
      reportError(error);
    }
  };

  if (catalog.error && !catalog.data) {
    return html`<${Notice} kind="danger" title="Couldn't load the built-in models">${catalog.error.message}<//>`;
  }
  if (!catalog.data) return html`<div class="panel-loading"><${Spinner} /> Loading models…</div>`;
  const statusText = statusLine(status);
  const dot = status?.state === "error" ? "dot-danger" : status?.state === "ready" ? "dot-success" : status?.state === "starting" ? "dot-warning dot-pulse" : "";
  return html`<div class="llm">
    ${statusText && html`<p class="llm-status"><span class=${`dot ${dot}`}></span>${statusText}</p>`}
    <ul class="llm-list">
      ${catalog.data.models.map((model) => {
        const inUse = polish.builtin_model === model.id && model.installed;
        const downloading = installing === model.id || progress?.id === model.id;
        const live = progress?.id === model.id ? progress : { done: 0, total: model.size || 0 };
        return html`<li class=${`llm-model${inUse ? " is-selected" : ""}`}>
          <div class="llm-main">
            <div class="llm-name">
              ${model.name}
              ${model.recommended && html`<${Badge} kind="accent">Recommended<//>`}
              ${inUse && html`<${Badge} kind="success" icon="check">In use<//>`}
            </div>
            <p class="llm-desc">${model.description}</p>
            <div class="llm-meta tnum">${formatBytes(model.size)} download${model.multilingual ? " · Many languages" : ""}</div>
            ${downloading &&
            html`<div class="llm-progress">
              <${Progress} done=${live.done} total=${live.total} indeterminate=${!live.done} label=${`Downloading ${model.name}`} />
              <${Button} kind="ghost" size="sm" onClick=${cancel}>Cancel<//>
            </div>`}
          </div>
          <div class="llm-actions">
            ${!model.installed && !downloading && html`<${Button} kind="secondary" size="sm" icon="download" disabled=${Boolean(installing)} onClick=${() => install(model)}>Download<//>`}
            ${model.installed && !inUse && html`<${Button} kind="secondary" size="sm" onClick=${() => saveSettings({ polish: { builtin_model: model.id } })}>Use<//>`}
            ${model.installed && html`<${IconButton} icon="trash-2" label="Remove model" tipSide="left" danger onClick=${() => uninstall(model)} />`}
          </div>
        </li>`;
      })}
    </ul>
  </div>`;
}

/** Models offered by the server at `url` (refetched whenever the address changes). */
function ModelPicker({ provider, url, value, onChange }) {
  const enabled = Boolean(url);
  const [state, setState] = useState({ models: null, error: null, loading: false });
  const load = async () => {
    setState((previous) => ({ ...previous, loading: true, error: null }));
    try {
      const result = await call("polish.models", { provider });
      setState({ models: result.models, error: null, loading: false });
    } catch (error) {
      setState({ models: null, error, loading: false });
    }
  };
  useEffect(() => {
    if (enabled) load();
    else setState({ models: null, error: null, loading: false });
  }, [provider, url]);

  const models = state.models || [];
  let prompt = "No models found";
  if (!enabled) prompt = "Add the server address first";
  else if (state.loading) prompt = "Looking for models…";
  else if (models.length) prompt = "Choose a model";
  const options = [{ value: "", label: prompt }, ...models.map((name) => ({ value: name, label: name }))];
  if (value && !models.includes(value)) options.push({ value, label: value });
  return html`<div class="picker">
    <div class="picker-line">
      <${Select}
        label="Model"
        value=${value}
        options=${options}
        onChange=${onChange}
        class="picker-select"
        disabled=${!enabled && !value}
      />
      <${IconButton} icon="refresh-cw" label="Refresh the list" onClick=${load} disabled=${state.loading || !enabled} />
    </div>
    ${state.error && html`<${FieldError}>${state.error.message}<//>`}
    ${enabled &&
    !state.loading &&
    !state.error &&
    state.models &&
    !models.length &&
    html`<span class="field-hint">Nothing came back. Check the address and that the server is running.</span>`}
  </div>`;
}

function ApiKeyField({ saved }) {
  const [editing, setEditing] = useState(!saved);
  const [draft, setDraft] = useState("");
  useEffect(() => setEditing(!saved), [saved]);
  const store = async (event) => {
    event.preventDefault();
    if (!draft.trim()) return;
    const ok = await saveSettings({ polish: { openai_api_key: draft.trim() } });
    if (ok) {
      setDraft("");
      setEditing(false);
      toast("API key saved. It's encrypted for your Windows account.", { kind: "success" });
    }
  };
  const remove = async () => {
    const ok = await saveSettings({ polish: { openai_api_key: "" } });
    if (ok) toast("API key removed.");
  };
  if (!editing) {
    return html`<div class="key-saved">
      <${Badge} kind="success" icon="lock">Saved<//>
      <${Button} kind="ghost" size="sm" onClick=${() => setEditing(true)}>Replace<//>
      <${Button} kind="danger-ghost" size="sm" onClick=${remove}>Remove<//>
    </div>`;
  }
  return html`<form class="key-form" onSubmit=${store}>
    <input
      class="input"
      type="password"
      value=${draft}
      placeholder="Paste your API key"
      aria-label="API key"
      autocomplete="off"
      spellcheck="false"
      onInput=${(event) => setDraft(event.currentTarget.value)}
    />
    <${Button} kind="secondary" type="submit" disabled=${!draft.trim()}>Save<//>
    ${saved && html`<${Button} kind="ghost" onClick=${() => setEditing(false)}>Cancel<//>`}
  </form>`;
}

function TestButton() {
  const [state, setState] = useState({ running: false, result: null });
  const run = async () => {
    setState({ running: true, result: null });
    try {
      const result = await call("polish.test");
      setState({ running: false, result });
    } catch (error) {
      setState({ running: false, result: { ok: false, message: error.message } });
    }
  };
  return html`<div class="test">
    <${Button} kind="secondary" icon="zap" busy=${state.running} disabled=${state.running} onClick=${run}>
      ${state.running ? "Testing…" : "Test polish"}
    <//>
    ${state.result &&
    html`<span class=${`test-result ${state.result.ok ? "is-ok" : "is-error"}`} role="status">
      <${Icon} name=${state.result.ok ? "circle-check" : "circle-alert"} size=${15} />${state.result.message}
    </span>`}
  </div>`;
}

function InstructionsField({ value }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return html`<div class="field instructions">
    <textarea
      class="textarea"
      rows="3"
      maxlength="2000"
      value=${draft}
      aria-label="Custom instructions"
      placeholder="For example: Use British spelling. Never use emoji."
      onInput=${(event) => setDraft(event.currentTarget.value)}
      onBlur=${() => draft !== value && saveSettings({ polish: { custom_instructions: draft } })}
    ></textarea>
    <span class="field-hint tnum">${draft.length} / 2,000</span>
  </div>`;
}

function PolishCard() {
  const polish = settings.value.polish;
  const enabled = polish.enabled;
  const set = (patch) => saveSettings({ polish: patch });
  return html`<${Card}
    class="polish-card"
    title="AI polish"
    icon="sparkles"
    description="A language model fixes grammar and wording after the basic cleanup. It also powers command mode."
    actions=${html`<${Toggle} label="AI polish" checked=${enabled} onChange=${(on) => set({ enabled: on })} />`}
  >
    ${enabled &&
    html`<div class="rows polish-rows">
      <${Row} label="Runs on" hint=${polish.provider === "builtin" ? "A small model that runs on this PC. Nothing leaves it." : polish.provider === "ollama" ? "Your own Ollama server." : "Any server that speaks the OpenAI API, like LM Studio. Text is sent to that server."}>
        <${Segmented}
          label="AI polish provider"
          value=${polish.provider}
          options=${[
            { value: "builtin", label: "Built-in" },
            { value: "ollama", label: "Ollama" },
            { value: "openai", label: "OpenAI-compatible" },
          ]}
          onChange=${(provider) => set({ provider })}
        />
      <//>
      ${polish.provider === "builtin" && html`<div class="row is-block"><${BuiltinPanel} polish=${polish} /></div>`}
      ${polish.provider === "ollama" &&
      html`
        <${Row} label="Server address">
          <${TextField} class="wide-field" label="Ollama server address" value=${polish.ollama_url} placeholder="http://127.0.0.1:11434" onCommit=${(url) => set({ ollama_url: url })} />
        <//>
        <${Row} label="Model" hint="Models you've pulled into Ollama.">
          <${ModelPicker} provider="ollama" url=${polish.ollama_url} value=${polish.ollama_model} onChange=${(model) => set({ ollama_model: model })} />
        <//>
      `}
      ${polish.provider === "openai" &&
      html`
        <${Row} label="Server address" hint="Include /v1, for example http://127.0.0.1:1234/v1.">
          <${TextField} class="wide-field" label="Server address" value=${polish.openai_url} placeholder="http://127.0.0.1:1234/v1" onCommit=${(url) => set({ openai_url: url })} />
        <//>
        <${Row} label="Model">
          <${ModelPicker} provider="openai" url=${polish.openai_url} value=${polish.openai_model} onChange=${(model) => set({ openai_model: model })} />
        <//>
        <${Row} label="API key" hint="Only if your server needs one. It's encrypted and never shown again.">
          <${ApiKeyField} saved=${polish.openai_api_key_saved} />
        <//>
      `}
      <${Row} label="Give up after" hint="If the model is slower than this, Whisp types your words without polish.">
        <${Slider}
          label="Polish time limit"
          min=${1}
          max=${60}
          step=${1}
          value=${Math.round(polish.timeout_seconds)}
          format=${(value) => `${value} s`}
          onCommit=${(value) => set({ timeout_seconds: value })}
        />
      <//>
      <div class="row is-top is-block">
        <div class="row-text">
          <div class="row-label">Custom instructions</div>
          <div class="row-hint">Extra guidance for every polish, in plain words.</div>
        </div>
        <${InstructionsField} value=${polish.custom_instructions} />
      </div>
      <div class="row is-block"><${TestButton} /></div>
    </div>`}
  <//>`;
}

export function StylesPage() {
  const text = settings.value.text;
  const version = `${text.remove_fillers}|${text.dictionary.length}|${text.dictionary.map((d) => d.term).join(",")}`;
  return html`<div class="page styles">
    <header class="page-header">
      <div>
        <h1 class="page-title">Styles</h1>
        <p class="page-desc">Choose how your words are written in each kind of app. Whisp picks the style from the app you're typing in.</p>
      </div>
    </header>
    <div class="stack-lg">
      <section>
        <div class="card styles-card">
          ${CATEGORIES.map(
            (category) => html`<${CategoryStyle} key=${category.id} category=${category} style=${text.styles[category.id]} version=${version} />`,
          )}
        </div>
        ${settings.value.polish.enabled && html`<p class="styles-note">Previews show the built-in cleanup. With AI polish on, wording gets a final touch too.</p>`}
      </section>

      <${PolishCard} />

      <${Card} title="Cleanup" description="Small fixes Whisp makes while you dictate.">
        <div class="rows">
          <${Row} label="Remove filler words" hint="Drops “um”, “uh” and false starts.">
            <${Toggle} label="Remove filler words" checked=${text.remove_fillers} onChange=${(on) => saveSettings({ text: { remove_fillers: on } })} />
          <//>
          <${Row} label="Voice commands" hint="Say “new line”, “new paragraph” or “scratch that” while dictating.">
            <${Toggle} label="Voice commands" checked=${text.voice_commands} onChange=${(on) => saveSettings({ text: { voice_commands: on } })} />
          <//>
        </div>
      <//>

      <${Card} title="App categories" description="Whisp sorts apps into the categories above. Move an app to a different one here.">
        <div class="card-body">
          <${AppOverrides}
            idPrefix="categories"
            value=${text.app_categories}
            options=${CATEGORIES.map((category) => ({ value: category.id, label: category.label }))}
            onChange=${(next) => saveSettings({ text: { app_categories: next } })}
          />
        </div>
      <//>
    </div>
  </div>`;
}
