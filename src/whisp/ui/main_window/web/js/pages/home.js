import { html } from "../html.js";
import { call } from "../bridge.js";
import { DayChart } from "../chart.js";
import { dayLabel, formatMinutes, formatNumber, formatTime, greeting, plural } from "../format.js";
import { useEvent, useResource, useThrottledCallback } from "../hooks.js";
import { Icon } from "../icons.js";
import { appStatus, engine, navigate, reportError, settings } from "../store.js";
import { Button, Card, Keycaps, Notice } from "../ui.js";

const todayFormat = new Intl.DateTimeFormat(undefined, { weekday: "long", day: "numeric", month: "long" });

function todayLabel() {
  return todayFormat.format(new Date());
}

function Greeting() {
  const info = useResource(() => call("app.info"), []);
  const name = info.data?.first_name;
  return html`<h1 class="page-title">${greeting()}${name ? `, ${name}` : ""}</h1>`;
}

export function HotkeyLine() {
  const chord = settings.value.hotkeys.dictate;
  if (!chord) {
    return html`<p class="hotkey-line">
      No dictation shortcut is set.
      <button class="link-btn" onClick=${() => navigate("settings", "shortcuts")}>Choose one</button>
    </p>`;
  }
  return html`<p class="hotkey-line">
    Hold <${Keycaps} chord=${chord} /> to dictate<span class="hotkey-sep" aria-hidden="true">·</span>tap to go hands-free
  </p>`;
}

function PausedNotice() {
  if (!appStatus.value.paused) return null;
  const resume = async () => {
    try {
      appStatus.value = await call("app.set_paused", { paused: false });
    } catch (error) {
      reportError(error);
    }
  };
  return html`<${Notice}
    icon="power"
    title="Whisp is paused"
    actions=${html`<${Button} kind="secondary" size="sm" onClick=${resume}>Resume<//>`}
  >
    Your shortcuts are off until you resume.
  <//>`;
}

function EngineNotice() {
  const status = engine.value;
  if (!status || status.state === "ready" || status.state === "loading" || status.state === "unloaded") return null;
  if (status.state === "not_installed") {
    return html`<${Notice}
      kind="warning"
      title="Download a speech model to start dictating"
      actions=${html`<${Button} kind="primary" size="sm" onClick=${() => navigate("settings", "model")}>Choose a model<//>`}
    >
      Whisp turns speech into text on this PC. It needs a model to do that.
    <//>`;
  }
  return html`<${Notice}
    kind="danger"
    title="The speech engine stopped"
    actions=${html`<${Button} kind="secondary" size="sm" onClick=${() => navigate("settings", "model")}>Speech settings<//>`}
  >
    ${status.message || "Whisp couldn't load the speech model."}
  <//>`;
}

function Stat({ icon, label, value, unit, sub, loading }) {
  return html`<div class="stat">
    <div class="stat-label"><${Icon} name=${icon} size=${15} />${label}</div>
    <div class=${`stat-value${loading ? " is-loading" : ""}`}>
      ${loading ? "—" : value}${unit && !loading && html`<span class="stat-unit">${unit}</span>`}
    </div>
    <div class="stat-sub">${loading ? " " : sub}</div>
  </div>`;
}

function Stats({ data }) {
  const loading = !data;
  const d = data || {};
  const streak = d.streak_days || 0;
  return html`<div class="stats">
    <${Stat}
      icon="audio-lines"
      label="Words this week"
      value=${formatNumber(d.words_this_week)}
      sub=${`${formatNumber(d.words_today)} today`}
      loading=${loading}
    />
    <${Stat}
      icon="gauge"
      label="Average speed"
      value=${formatNumber(d.avg_wpm)}
      unit="wpm"
      sub="Typing is about 40 wpm"
      loading=${loading}
    />
    <${Stat}
      icon="hourglass"
      label="Time saved"
      value=${formatMinutes(d.time_saved_minutes)}
      sub="Compared with typing"
      loading=${loading}
    />
    <${Stat}
      icon="flame"
      label="Streak"
      value=${formatNumber(streak)}
      unit=${streak === 1 ? "day" : "days"}
      sub=${`Best: ${plural(d.best_streak_days || 0, "day")}`}
      loading=${loading}
    />
  </div>`;
}

function TopApps({ apps }) {
  if (!apps.length) {
    return html`<p class="top-apps-empty">Apps you dictate into will show up here.</p>`;
  }
  const max = Math.max(...apps.map((app) => app.words), 1);
  return html`<ol class="top-apps">
    ${apps.map(
      (app) => html`<li class="top-app">
        <div class="top-app-line">
          <span class="top-app-name">${app.app || "Unknown app"}</span>
          <span class="top-app-words tnum">${formatNumber(app.words)}</span>
        </div>
        <div class="top-app-bar"><span style=${{ width: `${Math.max(2, (app.words / max) * 100)}%` }}></span></div>
      </li>`,
    )}
  </ol>`;
}

function Recent({ entries, historyOn }) {
  if (!historyOn) {
    return html`<p class="recent-note">History is off, so Whisp doesn't keep what you say. Turn it on in Settings → Privacy.</p>`;
  }
  if (!entries.length) return html`<p class="recent-note">Your latest dictations will show up here.</p>`;
  return html`<ul class="recent">
    ${entries.map(
      (entry) => html`<li class="recent-item">
        <span class="recent-time tnum">${dayLabel(entry.created_at) === "Today" ? formatTime(entry.created_at) : dayLabel(entry.created_at)}</span>
        <span class="recent-text">${entry.text || html`<span class="faint">No text saved</span>`}</span>
        <span class="recent-app">${entry.app_name}</span>
      </li>`,
    )}
  </ul>`;
}

function FirstRun() {
  const chord = settings.value.hotkeys.dictate;
  return html`<section class="card first-run">
    <div class="first-run-art" aria-hidden="true">
      <img src="/brand/logo-glyph.svg" alt="" />
    </div>
    <div class="first-run-body">
      <h2 class="first-run-title">Your first dictation is one shortcut away</h2>
      <ol class="first-run-steps">
        <li><span class="step-num">1</span><span>Click into any text box: an email, a chat, or the one below.</span></li>
        <li>
          <span class="step-num">2</span><span>Hold <${Keycaps} chord=${chord} /> and talk the way you normally would.</span>
        </li>
        <li><span class="step-num">3</span><span>Let go. Your words appear, cleaned up.</span></li>
      </ol>
      <textarea
        class="textarea practice"
        rows="3"
        placeholder="Click here, then hold the shortcut and say something."
        aria-label="Practice dictation"
      ></textarea>
    </div>
  </section>`;
}

export function HomePage() {
  const stats = useResource(() => call("stats.get"), []);
  const recent = useResource(() => call("history.list", { limit: 5 }), []);
  const refresh = useThrottledCallback(() => {
    stats.reload();
    recent.reload();
  }, 1000);
  useEvent("history.added", refresh);
  useEvent("history.changed", refresh);

  const data = stats.data;
  const fresh = data && data.total_dictations === 0;
  const historyOn = settings.value.privacy.history_enabled;

  return html`<div class="page home">
    <header class="page-header">
      <div>
        <p class="page-eyebrow">${todayLabel()}</p>
        <${Greeting} />
        <${HotkeyLine} />
      </div>
    </header>
    <div class="stack">
      <${PausedNotice} />
      <${EngineNotice} />
      ${stats.error && !data && html`<${Notice} kind="danger" title="Couldn't load your stats">${stats.error.message}<//>`}
      ${fresh
        ? html`<${FirstRun} />`
        : html`
            <${Stats} data=${data} />
            <div class="home-grid">
              <${Card}
                class="chart-card"
                title="Last 30 days"
                description=${data ? `${plural(data.daily.reduce((sum, day) => sum + day.words, 0), "word")} dictated` : " "}
              >
                <div class="card-body">${data ? html`<${DayChart} days=${data.daily} />` : html`<div class="chart-skeleton"></div>`}</div>
              <//>
              <${Card} title="Top apps" description="Where you dictate the most">
                <div class="card-body">${data && html`<${TopApps} apps=${data.top_apps} />`}</div>
              <//>
            </div>
            <${Card}
              title="Recent"
              actions=${html`<${Button} kind="ghost" size="sm" iconAfter="arrow-right" onClick=${() => navigate("history")}>
                View all
              <//>`}
            >
              <div class="card-body recent-body">${recent.data && html`<${Recent} entries=${recent.data.entries} historyOn=${historyOn} />`}</div>
            <//>
          `}
    </div>
  </div>`;
}
