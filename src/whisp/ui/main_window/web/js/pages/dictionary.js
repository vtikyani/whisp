import { useRef, useState } from "preact/hooks";
import { html } from "../html.js";
import { Icon } from "../icons.js";
import { saveSettings, settings, toast } from "../store.js";
import { Button, Empty, FieldError, IconButton } from "../ui.js";

const MAX_TERM = 100;

function parseVariants(text) {
  const seen = new Set();
  const out = [];
  for (const raw of text.split(/[,\n]/)) {
    const item = raw.trim();
    if (item && item.length <= MAX_TERM && !seen.has(item.toLowerCase())) {
      seen.add(item.toLowerCase());
      out.push(item);
    }
  }
  return out;
}

function validate(term, entries, ignoreIndex = -1) {
  const clean = term.trim();
  if (!clean) return "Type the word the way it should be written.";
  if (clean.length > MAX_TERM) return `Keep it under ${MAX_TERM} characters.`;
  const clash = entries.findIndex((entry, index) => index !== ignoreIndex && entry.term.toLowerCase() === clean.toLowerCase());
  if (clash !== -1) return `“${clean}” is already in your dictionary.`;
  return null;
}

function EntryEditor({ entry, index, entries, onDone }) {
  const [term, setTerm] = useState(entry.term);
  const [variants, setVariants] = useState(entry.sounds_like.join(", "));
  const [error, setError] = useState(null);
  const save = () => {
    const problem = validate(term, entries, index);
    if (problem) {
      setError(problem);
      return;
    }
    const next = [...entries];
    next[index] = { term: term.trim(), sounds_like: parseVariants(variants) };
    saveSettings({ text: { dictionary: next } });
    onDone();
  };
  const onKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      save();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      onDone();
    }
  };
  return html`<li class="entry is-editing">
    <div class="entry-form">
      <input
        class="input"
        value=${term}
        aria-label="Word"
        maxlength=${MAX_TERM}
        spellcheck="false"
        autofocus
        onInput=${(event) => setTerm(event.currentTarget.value)}
        onKeyDown=${onKeyDown}
      />
      <input
        class="input"
        value=${variants}
        aria-label="Sounds like, separated by commas"
        placeholder="Sounds like (separate with commas)"
        spellcheck="false"
        onInput=${(event) => setVariants(event.currentTarget.value)}
        onKeyDown=${onKeyDown}
      />
      <div class="entry-form-actions">
        <${Button} kind="ghost" size="sm" onClick=${onDone}>Cancel<//>
        <${Button} kind="primary" size="sm" onClick=${save}>Save<//>
      </div>
    </div>
    ${error && html`<${FieldError}>${error}<//>`}
  </li>`;
}

export function DictionaryPage() {
  const entries = settings.value.text.dictionary;
  const [term, setTerm] = useState("");
  const [variants, setVariants] = useState("");
  const [error, setError] = useState(null);
  const [editing, setEditing] = useState(null);
  const [filter, setFilter] = useState("");
  const termInput = useRef(null);

  const add = (event) => {
    event.preventDefault();
    const problem = validate(term, entries);
    if (problem) {
      setError(problem);
      return;
    }
    saveSettings({ text: { dictionary: [{ term: term.trim(), sounds_like: parseVariants(variants) }, ...entries] } });
    setTerm("");
    setVariants("");
    setError(null);
    setEditing(null);
    termInput.current?.focus();
  };

  const remove = (index) => {
    const removed = entries[index];
    setEditing(null);
    saveSettings({ text: { dictionary: entries.filter((_, i) => i !== index) } });
    toast(`Removed “${removed.term}”.`, {
      action: {
        label: "Undo",
        run: () => {
          const now = [...settings.value.text.dictionary];
          if (now.some((entry) => entry.term.toLowerCase() === removed.term.toLowerCase())) return;
          now.splice(Math.min(index, now.length), 0, removed);
          saveSettings({ text: { dictionary: now } });
        },
      },
    });
  };

  const needle = filter.trim().toLowerCase();
  const visible = entries
    .map((entry, index) => ({ entry, index }))
    .filter(
      ({ entry }) =>
        !needle || entry.term.toLowerCase().includes(needle) || entry.sounds_like.some((v) => v.toLowerCase().includes(needle)),
    );

  return html`<div class="page dictionary">
    <header class="page-header">
      <div>
        <h1 class="page-title">Dictionary</h1>
        <p class="page-desc">Teach Whisp names and words it gets wrong. When it hears a “sounds like” version, it writes your spelling.</p>
      </div>
    </header>

    <div class="stack">
      <form class="card add-card" onSubmit=${add}>
        <div class="add-fields">
          <label class="field">
            <span class="field-label">Word</span>
            <input
              class=${`input${error ? " is-invalid" : ""}`}
              ref=${termInput}
              value=${term}
              placeholder="e.g. Kubernetes"
              maxlength=${MAX_TERM}
              spellcheck="false"
              onInput=${(event) => {
                setTerm(event.currentTarget.value);
                setError(null);
              }}
            />
          </label>
          <label class="field">
            <span class="field-label">Sounds like <span class="faint">(optional)</span></span>
            <input
              class="input"
              value=${variants}
              placeholder="e.g. cooper netties, kubernetties"
              spellcheck="false"
              onInput=${(event) => setVariants(event.currentTarget.value)}
            />
          </label>
          <${Button} kind="primary" icon="plus" type="submit" class="add-button">Add<//>
        </div>
        ${error && html`<div class="add-error"><${FieldError}>${error}<//></div>`}
      </form>

      ${entries.length > 8 &&
      html`<span class="input-icon list-filter">
        <${Icon} name="search" />
        <input
          class="input"
          type="search"
          value=${filter}
          placeholder="Filter words"
          aria-label="Filter words"
          onInput=${(event) => setFilter(event.currentTarget.value)}
        />
      </span>`}

      ${entries.length === 0
        ? html`<section class="card">
            <${Empty} icon="book-a" title="No words yet">
              Add the names, brands and jargon you use, and Whisp will spell them your way every time.
            <//>
          </section>`
        : html`<ul class="card entries" aria-label="Dictionary words">
            ${visible.length === 0 && html`<li class="entries-none">Nothing matches “${filter.trim()}”.</li>`}
            ${visible.map(({ entry, index }) =>
              editing === index
                ? html`<${EntryEditor} key=${`edit-${entry.term}`} entry=${entry} index=${index} entries=${entries} onDone=${() => setEditing(null)} />`
                : html`<li class="entry" key=${entry.term}>
                    <div class="entry-main">
                      <span class="entry-term">${entry.term}</span>
                      ${entry.sounds_like.length > 0 &&
                      html`<span class="entry-variants">
                        <span class="entry-hint">sounds like</span>
                        ${entry.sounds_like.map((variant) => html`<span class="chip">${variant}</span>`)}
                      </span>`}
                    </div>
                    <span class="entry-actions">
                      <${IconButton} icon="pencil" label="Edit" onClick=${() => setEditing(index)} />
                      <${IconButton} icon="trash-2" label="Remove" tipSide="left" danger onClick=${() => remove(index)} />
                    </span>
                  </li>`,
            )}
          </ul>`}
    </div>
  </div>`;
}
