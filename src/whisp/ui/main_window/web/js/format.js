// Formatting helpers: numbers, sizes, durations, dates.

const numberFormat = new Intl.NumberFormat(undefined);
const compactFormat = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
const timeFormat = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });
const weekdayFormat = new Intl.DateTimeFormat(undefined, { weekday: "long" });
const dayMonthFormat = new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric", month: "short" });
const dayMonthYearFormat = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric" });
const shortDateFormat = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short" });

export function formatNumber(value) {
  return numberFormat.format(Math.round(value || 0));
}

export function formatCompact(value) {
  return value >= 10000 ? compactFormat.format(value) : formatNumber(value);
}

export function plural(count, one, many = `${one}s`) {
  return `${formatNumber(count)} ${count === 1 ? one : many}`;
}

export function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  if (bytes >= 1e3) return `${Math.round(bytes / 1e3)} KB`;
  return `${bytes} B`;
}

/** "2h 15m", "45m", "under a minute". */
export function formatMinutes(minutes) {
  const total = Math.max(0, Math.round(minutes || 0));
  if (total < 1) return "0m";
  const hours = Math.floor(total / 60);
  const rest = total % 60;
  if (hours === 0) return `${rest}m`;
  if (hours >= 100 || rest === 0) return `${formatNumber(hours)}h`;
  return `${hours}h ${rest}m`;
}

export function formatSeconds(ms) {
  const seconds = Math.max(0, Math.round((ms || 0) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

export function formatTime(timestamp) {
  return timeFormat.format(new Date(timestamp * 1000));
}

function startOfDay(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

/** A key for grouping timestamps by local day. */
export function dayKey(timestamp) {
  const date = new Date(timestamp * 1000);
  return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
}

/** "Today", "Yesterday", "Monday", "Sat 4 Oct", or with a year when it isn't this year. */
export function dayLabel(timestamp) {
  const date = new Date(timestamp * 1000);
  const today = startOfDay(new Date());
  const days = Math.round((today - startOfDay(date)) / 86400000);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days > 1 && days < 7) return weekdayFormat.format(date);
  if (date.getFullYear() === today.getFullYear()) return dayMonthFormat.format(date);
  return dayMonthYearFormat.format(date);
}

/** Parses "YYYY-MM-DD" as a local date. */
export function parseDay(text) {
  const [year, month, day] = text.split("-").map(Number);
  return new Date(year, month - 1, day);
}

export function formatDay(date, { weekday = false } = {}) {
  return weekday ? dayMonthFormat.format(date) : shortDateFormat.format(date);
}

const languageNames = (() => {
  try {
    return new Intl.DisplayNames(["en"], { type: "language" });
  } catch {
    return null;
  }
})();

export function languageName(code) {
  if (!code) return "";
  try {
    return (languageNames && languageNames.of(code)) || code;
  } catch {
    return code;
  }
}

export function greeting(date = new Date()) {
  const hour = date.getHours();
  if (hour < 5) return "Good evening";
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}
