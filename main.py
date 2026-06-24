"""Code Magic — real-time hand tracking with hacker/terminal visual effects."""

import sys
import time
import datetime
import cv2
import numpy as np

from tracker import HandTracker
from effects import (
    MatrixRainEffect,
    CodeTrailEffect,
    SyntaxExplosionEffect,
    BinaryPinchDrawEffect,
    TerminalSummonEffect,
    StackTraceCurseEffect,
    TwoHandCompileEffect,
    HUDEffect,
    scanline_overlay,
    draw_skeleton_code,
    detect_gesture,
    GREEN, AMBER, DIM_GREEN,
)

# ── Camera setup ──────────────────────────────────────────────────────────────

PREFERRED_W, PREFERRED_H = 1280, 720
FALLBACK_W,  FALLBACK_H  = 640, 480


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("       Check that a camera is connected and that this app has camera permissions.")
        print("       macOS: System Settings → Privacy & Security → Camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PREFERRED_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}×{h}")
    return cap, w, h


# ── Gesture → effect index mapping ───────────────────────────────────────────
#   0 MatrixRain  1 CodeTrail  2 SyntaxExplosion  3 BinaryPinch
#   4 Terminal    5 StackTrace 6 TwoHandCompile

GESTURE_MAP = {
    "none":       0,
    "custom":     0,
    "peace":      1,
    "fist":       2,
    "pinch":      3,
    "open_palm":  4,
    "point":      5,
    "two_hands":  6,
    "dual_palm":  6,
}

EFFECT_NAMES = [
    "matrix_rain", "code_trail", "syntax_explosion",
    "binary_pinch_draw", "terminal_summon", "stack_trace_curse",
    "two_hand_compile",
]


def print_guide():
    print("""
╔══════════════════════════════════════════════════╗
║            CODE MAGIC  //  KEYBOARD CONTROLS      ║
╠══════════════════════════════════════════════════╣
║  q  → quit                                       ║
║  c  → clear canvas  (// canvas cleared)          ║
║  s  → screenshot (screenshot_TIMESTAMP.png)      ║
║  a  → auto / gesture-driven mode                 ║
║  b  → toggle background (black / darkened webcam)║
║  1  → [force] matrix_rain                        ║
║  2  → [force] code_trail                         ║
║  3  → [force] syntax_explosion                   ║
║  4  → [force] binary_pinch_draw                  ║
║  5  → [force] terminal_summon                    ║
║  6  → [force] stack_trace_curse                  ║
║  7  → [force] two_hand_compile                   ║
╠══════════════════════════════════════════════════╣
║  GESTURES (auto mode)                            ║
║  ✌  peace sign     → code_trail                  ║
║  ☝  point only     → stack_trace_curse           ║
║  🤏 pinch           → binary_pinch_draw           ║
║  🖐  open palm      → terminal_summon             ║
║  ✊  fist           → syntax_explosion            ║
║  ✊→🖐 fist → open  → EXPLODE burst               ║
║  🙌 both palms     → two_hand_compile             ║
║  (default)         → matrix_rain                 ║
╚══════════════════════════════════════════════════╝
""")


def main():
    print_guide()

    cap, frame_w, frame_h = open_camera()
    tracker = HandTracker(max_hands=2)

    # Initialise all effects
    effects = [
        MatrixRainEffect(frame_w, frame_h),   # 0  key 1
        CodeTrailEffect(),                     # 1  key 2
        SyntaxExplosionEffect(),               # 2  key 3  (also always-updated)
        BinaryPinchDrawEffect(),               # 3  key 4
        TerminalSummonEffect(),                # 4  key 5
        StackTraceCurseEffect(),               # 5  key 6
        TwoHandCompileEffect(),                # 6  key 7
    ]
    explosion = effects[2]   # always updated so fist→open is never missed
    hud = HUDEffect()

    mode         = "auto"
    forced_idx   = 0
    active_idx   = 0
    bg_dark      = True      # False = pure black, True = darkened webcam

    fps_buf = []
    t_prev  = time.time()

    cv2.namedWindow("code magic", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("code magic", frame_w, frame_h)

    while True:
        ret, raw = cap.read()
        if not ret:
            print("WARNING: dropped frame")
            continue

        frame = cv2.flip(raw, 1)

        # Hand tracking
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = tracker.process(rgb)

        # Gesture + effect selection
        gesture = detect_gesture(hands)

        if mode == "auto":
            active_idx = GESTURE_MAP.get(gesture, 0)
        else:
            active_idx = forced_idx

        active_effect = effects[active_idx]

        # Update active effect + always track explosion gesture
        active_effect.update(hands, frame_w, frame_h)
        if active_effect is not explosion:
            explosion.update(hands, frame_w, frame_h)

        # ── Render pipeline ────────────────────────────────────────────────
        # 1. Base: pure black or darkened webcam
        if bg_dark:
            base = frame.copy()
        else:
            base = np.zeros_like(frame)

        # 2. Scanlines
        base = scanline_overlay(base, step=4, alpha=0.15)

        # 3. Active effect
        output = active_effect.draw(base)

        # 4. Explosion overlay if it has live particles (always visible)
        if active_effect is not explosion and explosion.has_content():
            output = explosion.draw(output)

        # 5. Hand skeleton
        for hand in hands:
            draw_skeleton_code(output, hand)

        # 6. FPS
        t_now = time.time()
        dt    = max(t_now - t_prev, 1e-6)
        t_prev = t_now
        fps_buf.append(1.0 / dt)
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = float(np.mean(fps_buf))

        # 7. HUD
        coords = (float(hands[0].palm_center[0] / frame_w),
                  float(hands[0].palm_center[1] / frame_h)) if hands else (0.0, 0.0)
        hud.set_info(gesture, EFFECT_NAMES[active_idx], fps,
                     mode if mode == "auto" else f"forced-{active_idx+1}", coords)
        output = hud.draw(output)

        cv2.imshow("code magic", output)

        # ── Key handling ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            for e in effects:
                e.clear()
            print("// canvas cleared")
        elif key == ord('a'):
            mode = "auto"
            print("// mode: auto (gesture-driven)")
        elif key == ord('b'):
            bg_dark = not bg_dark
            print(f"// background: {'darkened webcam' if bg_dark else 'pure black'}")
        elif key == ord('s'):
            ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_{ts}.png"
            cv2.imwrite(fname, output)
            print(f"// screenshot saved: {fname}")
        elif ord('1') <= key <= ord('7'):
            forced_idx = key - ord('1')
            mode = f"forced-{forced_idx + 1}"
            print(f"// forced effect: {EFFECT_NAMES[forced_idx]}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("// process exited cleanly")


if __name__ == "__main__":
    main()
