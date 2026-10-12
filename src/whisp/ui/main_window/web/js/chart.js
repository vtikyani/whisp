// Words per day for the last 30 days: thin rounded columns, today in opal.
import { useState } from "preact/hooks";
import { html } from "./html.js";
import { formatCompact, formatDay, formatNumber, parseDay, plural } from "./format.js";

function niceCeiling(value) {
  if (value <= 0) return 10;
  const power = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) {
    if (step * power >= value) return step * power;
  }
  return 10 * power;
}

function barHeight(words, ceiling) {
  return words > 0 ? Math.max(3, (words / ceiling) * 100) : 0;
}

export function DayChart({ days }) {
  const [active, setActive] = useState(null);
  const count = days.length;
  const today = count - 1;
  const peak = days.reduce((best, day, index) => (day.words > days[best].words ? index : best), 0);
  const ceiling = niceCeiling(days[peak]?.words || 0);
  const total = days.reduce((sum, day) => sum + day.words, 0);
  const dates = days.map((day) => parseDay(day.date));
  const middle = Math.floor(count / 2);

  const summary =
    total === 0
      ? "No words dictated in the last 30 days."
      : `Words per day for the last 30 days. Most on ${formatDay(dates[peak], { weekday: true })} with ${plural(
          days[peak].words,
          "word",
        )}. Today: ${plural(days[today].words, "word")}.`;

  const onKeyDown = (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1, Home: -count, End: count }[event.key];
    if (step === undefined) return;
    event.preventDefault();
    const from = active ?? today;
    setActive(Math.max(0, Math.min(today, from + step)));
  };

  return html`<figure class="chart">
    <div
      class="chart-plot"
      tabindex="0"
      role="img"
      aria-label=${summary}
      onMouseLeave=${() => setActive(null)}
      onBlur=${() => setActive(null)}
      onKeyDown=${onKeyDown}
    >
      <div class="chart-gridline is-top"><span class="chart-tick tnum">${formatCompact(ceiling)}</span></div>
      <div class="chart-gridline is-mid"><span class="chart-tick tnum">${formatCompact(ceiling / 2)}</span></div>
      <div class="chart-bars">
        ${days.map((day, index) => {
          const classes = ["chart-bar"];
          if (index === today) classes.push("is-today");
          if (day.words === 0) classes.push("is-zero");
          if (active === index) classes.push("is-active");
          return html`<div class="chart-slot" onMouseEnter=${() => setActive(index)}>
            <div class=${classes.join(" ")} style=${{ height: day.words > 0 ? `${barHeight(day.words, ceiling)}%` : undefined }}></div>
          </div>`;
        })}
        ${active !== null &&
        html`<div
          class=${`chart-tip ${active < 4 ? "is-start" : active > count - 5 ? "is-end" : ""}`}
          style=${{
            left: `${((active + 0.5) / count) * 100}%`,
            bottom: `calc(${barHeight(days[active].words, ceiling)}% + 10px)`,
          }}
          aria-hidden="true"
        >
          <span class="chart-tip-date">${active === today ? "Today" : formatDay(dates[active], { weekday: true })}</span>
          <span class="chart-tip-value tnum">${plural(days[active].words, "word")}</span>
        </div>`}
      </div>
    </div>
    <figcaption class="chart-axis" aria-hidden="true">
      <span>${formatDay(dates[0])}</span>
      <span>${formatDay(dates[middle])}</span>
      <span>Today</span>
    </figcaption>
    <table class="sr-only">
      <caption>Words per day</caption>
      <thead><tr><th scope="col">Date</th><th scope="col">Words</th></tr></thead>
      <tbody>
        ${days.map(
          (day, index) => html`<tr><td>${formatDay(dates[index], { weekday: true })}</td><td>${formatNumber(day.words)}</td></tr>`,
        )}
      </tbody>
    </table>
  </figure>`;
}
