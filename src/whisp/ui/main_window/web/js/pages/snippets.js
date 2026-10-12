import { useRef, useState } from "preact/hooks";
import { html } from "../html.js";
import { Icon } from "../icons.js";
import { saveSettings, settings, toast } from "../store.js";
import { Button, Empty, FieldError, IconButton } from "../ui.js";

const MAX_TRIGGER = 100;
const MAX_TEXT = 5000;

/** Multi-line text on one line, with a quiet ↵ where each line break goes. */
function previewLines(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  return lines.map((line, index) => html`${index > 0 && html`<span class="snippet-break" aria-hidden="true">↵</span>`}${line}`);
}

function validate(trigger, text, snippets, ignoreIndex = -1) {
  const cleanTrigger = trigger.trim();
  if (!cleanTrigger) return "Type the phrase you'll say.";
  if (cleanTrigger.length > MAX_TRIGGER) return `Keep the phrase under ${MAX_TRIGGER} characters.`;
  if (!text.trim()) return "Type the text Whisp should insert.";
  if (text.length > MAX_TEXT) return `Keep the text under ${MAX_TEXT.toLocaleString()} characters.`;
  const clash = snippets.findIndex(
    (snippet, index) => index !== ignoreIndex && snippet.trigger.toLowerCase() === cleanTrigger.toLowerCase(),
  );
  if (clash !== -1) return `You already have a snippet for “${cleanTrigger}”.`;
  return null;
}

function SnippetForm({ initial, submitLabel, onSubmit, onCancel, snippets, ignoreIndex = -1, autoFocus }) {
  const [trigger, setTrigger] = useState(initial?.trigger || "");
  const [text, setText] = useState(initial?.text || "");
  const [error, setError] = useState(null);
  const triggerInput = useRef(null);
  const submit = (event) => {
    event.preventDefault();
    const problem = validate(trigger, text, snippets, ignoreIndex);
    if (problem) {
      setError(problem);
      return;
    }
    onSubmit({ trigger: trigger.trim(), text });
    if (!initial) {
      setTrigger("");
      setText("");
      triggerInput.current?.focus();
    }
    setError(null);
  };
  return html`<form
    class="snippet-form"
    onSubmit=${submit}
    onKeyDown=${(event) => {
      if (event.key === "Escape" && onCancel) {
        event.preventDefault();
        onCancel();
      }
    }}
  >
    <label class="field">
      <span class="field-label">When you say</span>
      <input
        class="input"
        ref=${triggerInput}
        value=${trigger}
        placeholder="e.g. my calendar link"
        maxlength=${MAX_TRIGGER}
        spellcheck="false"
        autofocus=${autoFocus}
        onInput=${(event) => {
          setTrigger(event.currentTarget.value);
          setError(null);
        }}
      />
    </label>
    <label class="field">
      <span class="field-label">Whisp types</span>
      <textarea
        class="textarea"
        rows="2"
        value=${text}
        placeholder="e.g. https://cal.com/your-name/30min"
        maxlength=${MAX_TEXT}
        onInput=${(event) => {
          setText(event.currentTarget.value);
          setError(null);
        }}
        onKeyDown=${(event) => {
          if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) submit(event);
        }}
      ></textarea>
    </label>
    <div class="snippet-form-footer">
      ${error ? html`<${FieldError}>${error}<//>` : html`<span class="field-hint">Ctrl + Enter to save</span>`}
      <div class="snippet-form-actions">
        ${onCancel && html`<${Button} kind="ghost" onClick=${onCancel}>Cancel<//>`}
        <${Button} kind="primary" icon=${initial ? undefined : "plus"} type="submit">${submitLabel}<//>
      </div>
    </div>
  </form>`;
}

export function SnippetsPage() {
  const snippets = settings.value.text.snippets;
  const [editing, setEditing] = useState(null);
  const save = (next) => saveSettings({ text: { snippets: next } });

  const remove = (index) => {
    const removed = snippets[index];
    setEditing(null);
    save(snippets.filter((_, i) => i !== index));
    toast(`Removed “${removed.trigger}”.`, {
      action: {
        label: "Undo",
        run: () => {
          const now = [...settings.value.text.snippets];
          if (now.some((item) => item.trigger.toLowerCase() === removed.trigger.toLowerCase())) return;
          now.splice(Math.min(index, now.length), 0, removed);
          save(now);
        },
      },
    });
  };

  return html`<div class="page snippets">
    <header class="page-header">
      <div>
        <h1 class="page-title">Snippets</h1>
        <p class="page-desc">Say a short phrase and Whisp types the full text. Great for links, addresses and sign-offs.</p>
      </div>
    </header>

    <div class="stack">
      <section class="card add-card">
        <${SnippetForm} submitLabel="Add snippet" snippets=${snippets} onSubmit=${(item) => save([item, ...snippets])} />
      </section>

      ${snippets.length === 0
        ? html`<section class="card">
            <${Empty} icon="text-quote" title="No snippets yet">
              Add one above. Then say its phrase while dictating, and the full text appears instead.
            <//>
          </section>`
        : html`<ul class="card snippet-list" aria-label="Snippets">
            ${snippets.map((snippet, index) =>
              editing === index
                ? html`<li class="snippet is-editing" key=${`edit-${snippet.trigger}`}>
                    <${SnippetForm}
                      initial=${snippet}
                      submitLabel="Save"
                      snippets=${snippets}
                      ignoreIndex=${index}
                      autoFocus
                      onCancel=${() => setEditing(null)}
                      onSubmit=${(item) => {
                        const next = [...snippets];
                        next[index] = item;
                        save(next);
                        setEditing(null);
                      }}
                    />
                  </li>`
                : html`<li class="snippet" key=${snippet.trigger}>
                    <div class="snippet-main">
                      <div class="snippet-trigger">
                        <${Icon} name="audio-lines" size=${14} />
                        <span>“${snippet.trigger}”</span>
                      </div>
                      <p class="snippet-text selectable">${previewLines(snippet.text)}</p>
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
