// The wisp: three luminous opal ribbons that breathe when quiet and swell when you speak.
// Each ribbon is a sum of sines shaped by a bell envelope, so its ends taper to nothing.
import { useEffect, useRef } from "preact/hooks";
import { html } from "./html.js";
import { reducedMotion, theme } from "./store.js";

// The opal stops, squeezed inwards so both ends fade to nothing.
const STOPS = [
  [0, "rgba(103, 232, 249, 0)"],
  [0.1, "#67E8F9"],
  [0.38, "#818CF8"],
  [0.62, "#C084FC"],
  [0.9, "#F9A8D4"],
  [1, "rgba(249, 168, 212, 0)"],
];

// waves: [cycles across the width, phase speed (rad/s), weight]. twist offsets the second edge
// so the ribbon crosses itself like the logo. Back ribbons are softer and glow; the front one is crisp.
const RIBBONS = [
  { waves: [[1.15, 0.85, 1], [2.3, -1.3, 0.45]], twist: 2.1, gain: 1.05, fill: 0.16, line: 0.38, width: 2.4, glow: 12 },
  { waves: [[1.55, -1.05, 0.9], [2.85, 1.6, 0.35]], twist: 2.6, gain: 0.8, fill: 0.2, line: 0.5, width: 1.8, glow: 8 },
  { waves: [[1, 1.25, 1], [2.05, -0.75, 0.5], [3.4, 2.05, 0.2]], twist: 1.5, gain: 1.2, fill: 0.3, line: 1, width: 1.5, glow: 0 },
];

const SAMPLES = 96;
const ATTACK = 0.06;
const RELEASE = 0.25;

function shape(u, t, waves, offset) {
  let sum = 0;
  let norm = 0;
  for (const [cycles, speed, weight] of waves) {
    sum += Math.sin(u * cycles * Math.PI * 2 + t * speed + offset) * weight;
    norm += weight;
  }
  return sum / norm;
}

function envelope(u) {
  return Math.pow(Math.sin(Math.PI * u), 1.7);
}

function draw(ctx, width, height, t, level, light) {
  ctx.clearRect(0, 0, width, height);
  const centre = height / 2;
  const breathe = 0.5 + 0.5 * Math.sin(t * 1.15);
  const amplitude = height * (0.05 + 0.025 * breathe) + level * height * 0.34;
  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  for (const [offset, colour] of STOPS) gradient.addColorStop(offset, colour);

  for (const ribbon of RIBBONS) {
    const top = [];
    const bottom = [];
    for (let i = 0; i <= SAMPLES; i += 1) {
      const u = i / SAMPLES;
      const x = u * width;
      const env = envelope(u) * amplitude * ribbon.gain;
      top.push([x, centre + env * shape(u, t, ribbon.waves, 0)]);
      bottom.push([x, centre + env * 0.55 * shape(u, t * 0.9, ribbon.waves, ribbon.twist)]);
    }
    ctx.save();
    if (ribbon.glow && !light) {
      ctx.shadowColor = "rgba(167, 139, 250, 0.55)";
      ctx.shadowBlur = ribbon.glow;
    }
    ctx.beginPath();
    top.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    for (let i = bottom.length - 1; i >= 0; i -= 1) ctx.lineTo(bottom[i][0], bottom[i][1]);
    ctx.closePath();
    ctx.globalAlpha = ribbon.fill * (light ? 1.15 : 1);
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.beginPath();
    top.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.globalAlpha = ribbon.line;
    ctx.strokeStyle = gradient;
    ctx.lineWidth = ribbon.width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
    ctx.restore();
  }
}

// A slow, speech-like swell for decorative wisps that aren't tied to a microphone.
function demoLevel(seconds) {
  const phrase = 0.5 + 0.5 * Math.sin(seconds * 0.55);
  const syllables = 0.5 + 0.5 * Math.sin(seconds * 3.1) * Math.sin(seconds * 1.7);
  return 0.08 + 0.3 * phrase * syllables;
}

/**
 * level: a signal (0..1) from the microphone. Without one the wisp breathes on its
 * own: a calm hairline, or with `demo` a gentle swell for decorative use.
 * With reduced motion the ribbons hold still and fade with the level instead.
 */
export function Wisp({ level = null, demo = false, height = 96, label = "" }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let width = 0;
    let frame = 0;
    let smoothed = 0;
    let staticKey = "";
    let last = performance.now();
    const start = last;

    const resize = () => {
      const ratio = window.devicePixelRatio || 1;
      width = canvas.clientWidth;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      staticKey = "";
    };

    const tick = (now) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const target = level
        ? Math.max(0, Math.min(1, level.value))
        : demo
          ? demoLevel((now - start) / 1000)
          : 0;
      const tau = target > smoothed ? ATTACK : RELEASE;
      smoothed += (target - smoothed) * (1 - Math.exp(-dt / tau));
      const light = theme.value === "light";
      if (reducedMotion.value) {
        // Hold one still frame and let only the opacity follow the voice.
        const key = `${width}:${light}`;
        if (key !== staticKey) {
          draw(ctx, width, height, 0.6, 0.25, light);
          staticKey = key;
        }
        canvas.style.opacity = String(level ? 0.4 + 0.6 * Math.min(1, smoothed * 1.6) : 1);
      } else {
        staticKey = "";
        canvas.style.opacity = "1";
        draw(ctx, width, height, (now - start) / 1000, smoothed, light);
      }
      frame = document.hidden ? 0 : requestAnimationFrame(tick);
    };

    const onVisibility = () => {
      if (!document.hidden && !frame) {
        last = performance.now();
        frame = requestAnimationFrame(tick);
      }
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    document.addEventListener("visibilitychange", onVisibility);
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      frame = -1;
      observer.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [level, demo, height]);

  return html`<canvas
    class="wisp"
    ref=${canvasRef}
    style=${{ height: `${height}px` }}
    role=${label ? "img" : undefined}
    aria-label=${label || undefined}
    aria-hidden=${label ? undefined : "true"}
  ></canvas>`;
}
