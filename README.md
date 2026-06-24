# code_magic

Real-time hand tracking with programmer/hacker visual effects. Pure black terminal aesthetic, Matrix green, neon glow, CRT scanlines.

## Setup

```bash
cd code_magic

# Option A: reuse hand_magic's venv (same deps)
# ln -s ../hand_magic/.venv .venv

# Option B: fresh venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py
```

> **macOS camera permission**: System Settings → Privacy & Security → Camera → enable your terminal app.

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear all canvases (`// canvas cleared`) |
| `s` | Save timestamped screenshot |
| `a` | Auto / gesture-driven mode |
| `b` | Toggle background: pure black ↔ darkened webcam |
| `1` | Force matrix_rain |
| `2` | Force code_trail |
| `3` | Force syntax_explosion |
| `4` | Force binary_pinch_draw |
| `5` | Force terminal_summon |
| `6` | Force stack_trace_curse |
| `7` | Force two_hand_compile |

## Gestures (Auto Mode)

| Gesture | Effect |
|---------|--------|
| ✌ Peace (index + middle up) | **Code Trail** — rotated code tokens tumble from fingertip |
| ☝ Point (index only) | **Stack Trace Curse** — hold still 1 s to pour a cascading error waterfall |
| 🤏 Pinch (thumb + index) | **Binary Pinch Draw** — draw persistent binary `0101` lines |
| 🖐 Open palm (4–5 fingers) | **Terminal Summon** — fake terminal window types out commands |
| ✊ Fist | **Syntax Explosion** — open palm quickly for radial token burst |
| 🙌 Both palms open | **Two-Hand Compile** — progress bar between hands → BUILD SUCCESS / SEGFAULT |
| (none / default) | **Matrix Rain** — hand landmarks accelerate falling character columns |

## Visual Style

- **Background**: pure black (default) or 18% darkened webcam (`b` to toggle)
- **Primary**: Matrix green `(0, 255, 70)` + Cyan + Amber + Hot pink
- **Glow**: Gaussian blur additive blend on all effect layers
- **Scanlines**: every 4th row darkened 15% for CRT look
- **Skeleton**: `[+]` crosshairs at each landmark, dashed connections

## Architecture

| File | Role |
|------|------|
| `tracker.py` | `HandTracker` — MediaPipe wrapper with EMA smoothing, finger states, pinch, velocity |
| `effects.py` | 7 `Effect` subclasses + `HUDEffect` + drawing helpers (scanlines, dashed lines, PIL rotated text) |
| `main.py` | Camera loop, gesture routing, render pipeline, keyboard handling |
