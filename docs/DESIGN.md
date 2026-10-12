# Whisp design language

Whisp should feel calm, precise and a little magical. Think of a premium
utility like Linear, Raycast or Arc, not a busy dashboard. The UI stays quiet
and neutral. The one expressive element is the **wisp**: a thin, luminous
ribbon of breath that comes alive when you speak.

## Principles

- **Quiet by default, alive when listening.** Idle UI is monochrome. Colour and
  motion are reserved for voice: the wisp, the active state, the primary action.
- **Never in the way.** The overlay never takes focus, never covers what you're
  typing into, and gets out of the way the instant it's done.
- **Fast feels calm.** Motion is short and springy. Nothing bounces for show.
- **Readable first.** Contrast meets WCAG AA. Text never sits on the gradient.

## Colour

### The opal gradient

The signature colour. Use it for the wisp, the logo, the active nav marker, the
primary button's glow, and progress. Nowhere else.

| Stop | Hex |
|---|---|
| 0% | `#67E8F9` (ice) |
| 35% | `#818CF8` (periwinkle) |
| 65% | `#C084FC` (orchid) |
| 100% | `#F9A8D4` (blush) |

CSS: `linear-gradient(100deg, #67E8F9 0%, #818CF8 35%, #C084FC 65%, #F9A8D4 100%)`

The **solid accent** for controls (toggles, focus rings, links, selected
states) is periwinkle: `#8B9BFF` on dark, `#5B63E6` on light.

### Neutrals

| Token | Dark | Light |
|---|---|---|
| `bg` | `#0B0B0E` | `#F7F7F9` |
| `surface-1` (cards) | `#121216` | `#FFFFFF` |
| `surface-2` (raised, inputs) | `#18181E` | `#F1F1F5` |
| `surface-3` (hover) | `#202028` | `#E9E9EF` |
| `border` | `rgba(255,255,255,0.07)` | `rgba(12,12,20,0.08)` |
| `border-strong` | `rgba(255,255,255,0.12)` | `rgba(12,12,20,0.14)` |
| `text-1` | `#F4F4F6` | `#0E0E12` |
| `text-2` | `#B3B3BD` | `#4A4A55` |
| `text-3` | `#7B7B86` | `#8A8A96` |

### Status

| Token | Dark | Light |
|---|---|---|
| `success` | `#4ADE80` | `#16A34A` |
| `warning` | `#FBBF24` | `#D97706` |
| `danger` | `#F87171` | `#DC2626` |

## Type

- **Inter** (variable, bundled at `src/whisp/assets/fonts/InterVariable.ttf`, OFL).
  Used everywhere: web UI and Qt-painted overlay.
- Numbers use tabular figures (`font-feature-settings: "tnum"`).
- Scale (px): 12 · 13 · 14 (body) · 16 · 20 · 28 · 40 (hero numbers).
- Weights: 400 body, 500 labels/buttons, 600 headings. Headings use
  `letter-spacing: -0.015em`.
- Key caps (hotkeys) use Inter 500 at 12px in a raised `surface-2` cap with a
  1px `border-strong` and a 1px bottom inset shadow.

## Shape & space

- 4 px grid. Common gaps: 8, 12, 16, 24, 32.
- Radius: 8 controls · 12 cards · 16 panels · 999 pills.
- Shadows (dark): `0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px rgba(0,0,0,0.35)`.
  Light: `0 1px 2px rgba(16,16,24,0.06), 0 8px 24px rgba(16,16,24,0.08)`.

## Motion

- Durations: 120 ms (hover, press) · 200 ms (state change) · 320 ms (panels, the overlay morph).
- Easing: `cubic-bezier(0.2, 0.8, 0.2, 1)` for most things. The overlay morphs
  with a critically damped spring (no overshoot on size; at most 2% on scale).
- Honour Windows "Show animations in Windows" (reduced motion): fade only, no movement.

## The overlay (the pill)

A dark glass capsule, bottom-centre of the active monitor, 24 px above the
taskbar. Fill `rgba(14,14,18,0.86)`, 1 px border `rgba(255,255,255,0.10)`,
soft shadow, and a faint opal outer glow that breathes with your voice.

| State | Size (logical px) | Content |
|---|---|---|
| Idle (optional) | 40 × 6 | Hairline capsule at 35% opacity. On hover it grows to 168 × 30 and shows "Hold **Ctrl + Win** to dictate". Click starts hands-free. |
| Listening (hold) | 184 × 44 | The wisp. |
| Listening (hands-free) | 264 × 44 | ✕ cancel · the wisp · timer · ■ stop. Buttons never take focus. |
| Command mode | as above | A small "✦ Edit" chip at the left, tinted orchid, so it's obvious you're instructing. |
| Live preview | +28 high | One line of your words beneath the wisp, 13 px `text-2`, fading out at the left edge. |
| Processing | 120 × 44 | The wisp settles into a thin line and an opal shimmer sweeps across it. |
| Done | 44 × 44 | A check draws itself in `success`, then the pill shrinks away (≈ 600 ms total). |
| Copied | auto width | "Copied — press Ctrl + V to paste" for 2.5 s. |
| No speech | auto width | "Didn't catch that" for 1.6 s with a gentle 2-cycle shake. |
| Error | auto width | Danger-tinted border plus a short message. |

**The wisp** is three overlapping ribbons drawn with the opal gradient. Each is
a smooth curve made of a sum of sines with its own phase speed, shaped by a
bell envelope so the ends taper to nothing. Amplitude follows the mic level
(fast attack ≈ 60 ms, slow release ≈ 250 ms). When you're silent it's a calm,
slowly undulating hairline that breathes. When you speak it swells. The back
ribbons use lower alpha and a soft blur-like glow, the front ribbon a crisp
1.5 px core. 60 fps while visible; the animation timer stops when hidden.

## The main window

- Default 1080 × 720, minimum 900 × 600. Native Windows frame with the title
  bar coloured to match `bg` (dark or light).
- **Sidebar** (232 px): logo + wordmark, then Home, History, Dictionary,
  Snippets, Styles, Settings. The active item has a 2 px opal bar and
  `surface-2` fill. At the bottom, a status chip shows the engine state, e.g.
  "Parakeet · RTX 4070", with a green, amber or red dot.
- **Content**: page title (28 px, 600) with a one-line description in `text-2`,
  then cards on `surface-1`. Max content width 880 px.
- **Home**: greeting, the hotkey shown as key caps, four stat cards (words this
  week, average WPM, time saved, streak), a 30-day words chart (thin rounded
  bars, opal gradient on the current day), and recent dictations.
- **Empty states** teach: a sentence and the one action that fills them.
- **Onboarding** is full-window: Welcome → Microphone (live level meter) →
  Model (one-click download with progress, GPU badge) → Hotkey (record it, then
  practise in a text box) → Done.

## Icons & logo

- UI icons: Lucide (ISC licence), 1.75 px stroke, 16/18/20 px.
- Logo: `src/whisp/assets/brand/logo.svg` — a single opal wisp stroke on an
  ink rounded square. The tray icon uses the same stroke without the tile:
  monochrome white when idle, opal while recording.

## Writing

- Plain, short, friendly. Sentence case everywhere.
- Say what happened and what to do: "Microphone not found. Plug one in or pick
  another in Settings."
- Name keys the way Windows does: Ctrl, Shift, Alt, Win, Esc.
