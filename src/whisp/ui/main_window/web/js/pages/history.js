import { useEffect, useRef, useState } from "preact/hooks";
import { html } from "../html.js";
import { call } from "../bridge.js";
import { dayKey, dayLabel, formatSeconds, formatTime, plural } from "../format.js";
import { useDebounced, useEvent, useResource, useThrottledCallback } from "../hooks.js";
import { Icon } from "../icons.js";
import { reportError, saveSettings, settings, toast } from "../store.js";
import { Badge, Button, Empty, IconButton, Keycaps, Notice, Select, Spinner, confirmDialog } from "../ui.js";

const PAGE_SIZE = 40;

function groupByDay(entries) {
  const groups = [];
  for (const entry of entries) {
    const key = dayKey(entry.created_at);
    let group = groups[groups.length - 1];
    if (!group || group.key !== key) {
      group = { key, label: dayLabel(entry.created_at), items: [] };
      groups.push(group);
    }
    group.items.push(entry);
  }
  return groups;
}

function HistoryItem({ entry, expanded, onToggle, onCopy, onDelete }) {
  const toggle = () => {
    if (String(window.getSelection())) return; // the user is selecting text, not expanding
    onToggle();
  };
  const wpm = entry.speech_ms > 0 ? Math.round(entry.word_count / (entry.speech_ms / 60000)) : 0;
  return html`<li class=${`h-item${expanded ? " is-expanded" : ""}`}>
    <span class="h-time tnum">${formatTime(entry.created_at)}</span>
    <div
      class="h-body"
      role="button"
      tabindex="0"
      aria-expanded=${expanded ? "true" : "false"}
      onClick=${toggle}
      onKeyDown=${(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onToggle();
        }
      }}
    >
      <div class="h-meta">
        <span class="h-app">${entry.app_name || "Unknown app"}</span>
        <span class="h-sep" aria-hidden="true">·</span>
        <span class="tnum">${plural(entry.word_count, "word")}</span>
        ${entry.kind === "command" && html`<${Badge} icon="sparkles">Edit command<//>`}
        ${entry.polished && entry.kind !== "command" && html`<${Badge} icon="sparkles">Polished<//>`}
      </div>
      <p class="h-text selectable">${entry.text || html`<span class="faint">Text wasn't saved (history was off)</span>`}</p>
      ${expanded &&
      html`<div class="h-details">
        ${entry.raw_text && html`<div class="h-raw selectable"><span class="h-raw-label">What you said</span>${entry.raw_text}</div>`}
        <div class="h-facts tnum">
          ${entry.audio_ms > 0 && html`<span>Spoke for ${formatSeconds(entry.audio_ms)}</span>`}
          ${wpm > 0 && html`<span>${wpm} wpm</span>`}
          ${entry.latency_ms > 0 && html`<span>Ready in ${(entry.latency_ms / 1000).toFixed(1)}s</span>`}
        </div>
      </div>`}
    </div>
    <span class="h-actions">
      <${IconButton} icon="copy" label="Copy" onClick=${onCopy} disabled=${!entry.text} />
      <${IconButton} icon="trash-2" label="Delete" tipSide="left" danger onClick=${onDelete} />
    </span>
  </li>`;
}

function Skeleton() {
  return html`<ul class="h-list is-skeleton" aria-hidden="true">
    ${[0, 1, 2, 3].map(() => html`<li class="h-item"><span class="sk sk-time"></span><span class="sk sk-text"></span></li>`)}
  </ul>`;
}

export function HistoryPage() {
  const prefs = settings.value;
  const [query, setQuery] = useState("");
  const search = useDebounced(query.trim(), 220);
  const [app, setApp] = useState("");
  const [state, setState] = useState({
    entries: [],
    hasMore: false,
    loading: true,
    error: null,
    loaded: false,
    persistent: true,
  });
  const [expanded, setExpanded] = useState(() => new Set());
  const apps = useResource(() => call("history.apps"), []);
  const token = useRef(0);
  const current = useRef(state);
  current.current = state;
  const sentinel = useRef(null);

  const fetchPage = async ({ offset, limit, replace }) => {
    const id = ++token.current;
    setState((previous) => ({ ...previous, loading: true }));
    try {
      const result = await call("history.list", { query: search, app, limit, offset });
      if (id !== token.current) return;
      setState((previous) => ({
        entries: replace ? result.entries : [...previous.entries, ...result.entries],
        hasMore: result.has_more,
        loading: false,
        error: null,
        loaded: true,
        persistent: result.persistent !== false,
      }));
    } catch (error) {
      if (id === token.current) setState((previous) => ({ ...previous, loading: false, error, loaded: true }));
    }
  };

  useEffect(() => {
    fetchPage({ offset: 0, limit: PAGE_SIZE, replace: true });
  }, [search, app]);

  const refresh = useThrottledCallback(() => {
    fetchPage({ offset: 0, limit: Math.max(PAGE_SIZE, current.current.entries.length), replace: true });
    apps.reload();
  }, 600);
  useEvent("history.added", refresh);
  useEvent("history.changed", refresh);

  useEffect(() => {
    const target = sentinel.current;
    if (!target) return undefined;
    const observer = new IntersectionObserver(
      ([entry]) => {
        const now = current.current;
        if (entry.isIntersecting && now.hasMore && !now.loading) {
          fetchPage({ offset: now.entries.length, limit: PAGE_SIZE, replace: false });
        }
      },
      { root: target.closest(".main"), rootMargin: "400px" },
    );
    observer.observe(target);
    return () => observer.disconnect();
  }, [state.hasMore, state.loading, search, app]);

  const toggle = (id) =>
    setExpanded((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const copy = async (entry) => {
    try {
      await call("history.copy", { id: entry.id });
      toast("Copied to the clipboard.", { kind: "success" });
    } catch (error) {
      reportError(error);
    }
  };

  const remove = async (entry) => {
    setState((previous) => ({ ...previous, entries: previous.entries.filter((item) => item.id !== entry.id) }));
    try {
      await call("history.delete", { id: entry.id });
      toast("Dictation deleted.");
    } catch (error) {
      reportError(error);
      refresh();
    }
  };

  const clearAll = async () => {
    const ok = await confirmDialog({
      title: "Clear all history?",
      text: "This permanently deletes every saved dictation on this PC.",
      confirmLabel: "Clear history",
      danger: true,
    });
    if (!ok) return;
    try {
      await call("history.clear");
      setState((previous) => ({ ...previous, entries: [], hasMore: false, loading: false, error: null, loaded: true }));
      toast("History cleared.");
    } catch (error) {
      reportError(error);
    }
  };

  const retention = prefs.privacy.history_retention_days;
  const description = !prefs.privacy.history_enabled
    ? "Your saved dictations. History is off, so new ones aren't kept."
    : retention > 0
      ? `What you've dictated in the last ${plural(retention, "day")}. It never leaves this PC.`
      : "Everything you've dictated. It never leaves this PC.";

  const filtered = Boolean(search || app);
  const appOptions = [{ value: "", label: "All apps" }, ...(apps.data?.apps || []).map((name) => ({ value: name, label: name }))];
  const groups = groupByDay(state.entries);

  return html`<div class="page history">
    <header class="page-header">
      <div>
        <h1 class="page-title">History</h1>
        <p class="page-desc">${description}</p>
      </div>
      <div class="page-actions">
        <${Button} kind="ghost" icon="trash-2" onClick=${clearAll} disabled=${!state.entries.length && !filtered}>Clear all<//>
      </div>
    </header>

    <div class="stack">
      ${!prefs.privacy.history_enabled &&
      html`<${Notice}
        icon="eye-off"
        title="History is off"
        actions=${html`<${Button} kind="secondary" size="sm" onClick=${() => saveSettings({ privacy: { history_enabled: true } })}>
          Turn on
        <//>`}
      >
        Whisp still counts words for your stats, but forgets what you said.
      <//>`}

      <div class="toolbar">
        <span class="input-icon toolbar-search">
          <${Icon} name="search" />
          <input
            class="input"
            type="search"
            value=${query}
            placeholder="Search your dictations"
            aria-label="Search history"
            spellcheck="false"
            onInput=${(event) => setQuery(event.currentTarget.value)}
            onKeyDown=${(event) => event.key === "Escape" && setQuery("")}
          />
        </span>
        <${Select} label="Filter by app" value=${app} options=${appOptions} onChange=${setApp} class="toolbar-filter" />
      </div>

      ${!state.persistent &&
      html`<${Notice} kind="warning" title="History isn't being saved right now">
        Whisp couldn't open its history file, so this session's dictations will be gone when Whisp closes.
      <//>`}

      ${state.error && html`<${Notice} kind="danger" title="Couldn't load your history">${state.error.message}<//>`}

      ${!state.loaded
        ? html`<${Skeleton} />`
        : groups.length === 0
          ? html`<section class="card">
              ${filtered
                ? html`<${Empty}
                    icon="search"
                    title="No matches"
                    action=${html`<${Button}
                      kind="secondary"
                      onClick=${() => {
                        setQuery("");
                        setApp("");
                      }}
                      >Clear search<//
                    >`}
                    >Nothing ${search ? `matches “${search}”` : "here"}${app ? ` in ${app}` : ""}.<//
                  >`
                : html`<${Empty} icon="history" title="No dictations yet">
                    Hold <${Keycaps} chord=${prefs.hotkeys.dictate} /> in any app and speak. What you say shows up here.
                  <//>`}
            </section>`
          : groups.map(
              (group) => html`<section class="h-group" key=${group.key}>
                <h2 class="h-day">${group.label}</h2>
                <ul class="h-list card">
                  ${group.items.map(
                    (entry) => html`<${HistoryItem}
                      key=${entry.id}
                      entry=${entry}
                      expanded=${expanded.has(entry.id)}
                      onToggle=${() => toggle(entry.id)}
                      onCopy=${() => copy(entry)}
                      onDelete=${() => remove(entry)}
                    />`,
                  )}
                </ul>
              </section>`,
            )}
      <div ref=${sentinel} class="h-sentinel">${state.loading && state.loaded && state.hasMore && html`<${Spinner} />`}</div>
    </div>
  </div>`;
}
