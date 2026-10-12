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

/** `value` with `digits` decimals (0 or 1), sending exact ties to the even digit as Python's format() does. */
function fixed(value, digits) {
  const quarters = value * 4; // exact: a power of two
  const tie = digits === 0 ? quarters % 4 === 2 : Number.isInteger(quarters) && quarters % 2 === 1;
  if (!tie) return value.toFixed(digits);
  const scale = 10 ** digits;
  const down = Math.floor(value * scale); // exact for these values
  return ((down % 2 === 0 ? down : down + 1) / scale).toFixed(digits);
}

/**
 * "780 bytes", "63.4 MB", "631 MB", "1.1 GB": 1024-based, as Windows Explorer shows sizes, and written
 * exactly like Whisp's Python side writes them (whisp.core.asr.downloads.format_size), so a download
 * reads the same in a message and on its button.
 */
export function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  let value = Math.max(0, bytes);
  if (value < 1024) return `${Math.trunc(value)} bytes`;
  for (const unit of ["KB", "MB", "GB", "TB"]) {
    value /= 1024;
    if (value < 1024 || unit === "TB") return `${fixed(value, value < 100 ? 1 : 0)} ${unit}`;
  }
  return "";
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
