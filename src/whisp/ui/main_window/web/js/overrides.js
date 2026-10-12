// A list of per-app overrides: process name -> choice. Used for app categories and insertion methods.
import { useState } from "preact/hooks";
import { html } from "./html.js";
import { call } from "./bridge.js";
import { useResource } from "./hooks.js";
import { Button, FieldError, IconButton, Select } from "./ui.js";

function normaliseProcess(text) {
  let name = text.trim().toLowerCase();
  const match = name.match(/\(([^)]+\.exe)\)$/);
  if (match) name = match[1];
  if (name && !name.includes(".")) name = `${name}.exe`;
  return name;
}

export function AppOverrides({ value, options, onChange, idPrefix, placeholder = "App, e.g. notion.exe" }) {
  const known = useResource(() => call("history.processes"), []);
  const [draft, setDraft] = useState("");
  const [choice, setChoice] = useState(options[0].value);
  const [error, setError] = useState(null);
  const apps = known.data?.apps || [];
  const names = Object.fromEntries(apps.map((item) => [item.process, item.app]));
  const entries = Object.entries(value).sort(([a], [b]) => a.localeCompare(b));

  const add = () => {
    const process = normaliseProcess(draft);
    if (!process) {
      setError("Type the app's program name, like slack.exe.");
      return;
    }
    if (process.length > 100 || /[\\/:*?"<>|]/.test(process)) {
      setError("That doesn't look like a program name.");
      return;
    }
    setError(null);
    setDraft("");
    onChange({ ...value, [process]: choice });
  };

  const remove = (process) => {
    const { [process]: _removed, ...rest } = value;
    onChange(rest);
  };

  const listId = `${idPrefix}-apps`;
  return html`<div class="overrides">
    ${entries.length > 0 &&
    html`<ul class="overrides-list">
      ${entries.map(
        ([process, selected]) => html`<li class="override">
          <div class="override-app">
            <span class="override-name">${names[process] || process.replace(/\.exe$/, "")}</span>
            <span class="override-process">${process}</span>
          </div>
          <${Select}
            size="sm"
            label=${`Choice for ${process}`}
            value=${selected}
            options=${options}
            onChange=${(next) => onChange({ ...value, [process]: next })}
          />
          <${IconButton} icon="x" label="Remove" tipSide="left" onClick=${() => remove(process)} />
        </li>`,
      )}
    </ul>`}
    <form
      class="override-add"
      onSubmit=${(event) => {
        event.preventDefault();
        add();
      }}
    >
      <input
        class="input"
        list=${listId}
        value=${draft}
        placeholder=${placeholder}
        aria-label="App program name"
        spellcheck="false"
        maxlength="120"
        onInput=${(event) => setDraft(event.currentTarget.value)}
      />
      <datalist id=${listId}>
        ${apps.filter((item) => !(item.process in value)).map((item) => html`<option value=${item.process}>${item.app}</option>`)}
      </datalist>
      <${Select} label="Choice for the new app" value=${choice} options=${options} onChange=${setChoice} />
      <${Button} kind="secondary" icon="plus" type="submit" disabled=${!draft.trim()}>Add<//>
    </form>
    ${error && html`<${FieldError}>${error}<//>`}
  </div>`;
}
