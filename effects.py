"""Code Magic visual effects — hacker / terminal aesthetic."""

import cv2
import numpy as np
import time
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from tracker import HandData, FINGERTIP_IDS, INDEX_TIP, THUMB_TIP, HAND_CONNECTIONS

# ── Palette (BGR) ─────────────────────────────────────────────────────────────
GREEN     = (0, 255, 70)
DIM_GREEN = (0, 80, 20)
BRIGHT_G  = (180, 255, 180)
CYAN      = (0, 255, 255)
AMBER     = (0, 176, 255)   # BGR for amber (R=255 G=176 B=0)
PINK      = (200, 0, 255)   # BGR for hot pink
RED       = (40, 40, 220)
WHITE     = (200, 200, 200)
BLACK     = (0, 0, 0)
DIM_CYAN  = (0, 100, 100)

SYNTAX_COLORS = [GREEN, CYAN, AMBER, PINK, (100, 255, 100), (255, 100, 220)]

# ── Character / token sets ────────────────────────────────────────────────────
MATRIX_CHARS = list(
    "01010101ABCDEFabcdef{}[]();=></*+-|&@#$%^!?~`\\"
    "アイウエオカキクケコサシスセソタチツテトナニヌネノ"
)

CODE_TOKENS = [
    "if", "else", "def", "return", "class", "import", "from",
    "for", "while", "try", "catch", "null", "None", "True",
    "False", "=>", "&&", "||", "{}", "[]", "()", "::", "++",
    "--", "!=", "===", "<=", ">=", "async", "await", "const",
    "let", "var", "lambda", "yield", "self", "break", "raise",
    "0x00", "0xFF", "NaN", "Inf", "#!", "~>", ":=",
]

EXPLOSION_TOKENS = [
    "print()", "[0]", "git push", "sudo", "NullPointerException",
    "undefined", "{ }", "===", "//TODO", "segfault", "404",
    "rm -rf /", "malloc()", "NULL", "panic!", "SIGKILL",
    "git blame", "node_modules/", "LGTM", "yolo",
    "TypeError", "buffer overflow", "∞ loop", "kernel panic",
    "stack overflow", "out of memory", "syntax error",
]

COMMANDS = [
    "npm install",
    'python main.py',
    'git commit -m "magic"',
    "sudo rm -rf /",
    "Hello, World!",
    "make install",
    "cargo build --release",
    "go run main.go",
    "docker ps -a",
    "kubectl apply -f .",
    "pip freeze > reqs.txt",
    "chmod 777 universe",
    "gcc -o magic main.c && ./magic",
    "./configure --prefix=/usr/local",
]

STACK_TRACE = [
    "Traceback (most recent call last):",
    '  File "reality.py", line 42, in universe',
    '    result = magic.cast(spell, target=self)',
    '  File "magic.py", line 7, in cast',
    '    return self._invoke(args, force=True)',
    '  File "magic.py", line 23, in _invoke',
    '    if not physics.check_laws(): raise',
    '  File "physics.py", line 1337, in check',
    '    assert self.entropy >= 0, "heat death"',
    'AssertionError: heat death',
    'RuntimeError: reality.exe has stopped working',
    'Segmentation fault (core dumped)',
    'Process finished with exit code 139',
    '> core dumped to /dev/null',
]

# ── Font loader ───────────────────────────────────────────────────────────────
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

def _load_mono_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]
    font = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


# ── Drawing helpers ───────────────────────────────────────────────────────────

def additive_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    return np.clip(base.astype(np.int32) + overlay.astype(np.int32), 0, 255).astype(np.uint8)


def glow_blur(layer: np.ndarray, ksize: int = 9) -> np.ndarray:
    k = ksize | 1
    blurred = cv2.GaussianBlur(layer, (k, k), 0)
    return np.clip(layer.astype(np.int32) + blurred.astype(np.int32), 0, 255).astype(np.uint8)


def scanline_overlay(frame: np.ndarray, step: int = 4, alpha: float = 0.15) -> np.ndarray:
    """Darken every `step`th row for CRT scanline look."""
    out = frame.copy()
    out[::step] = (out[::step] * (1.0 - alpha)).astype(np.uint8)
    return out


def draw_crosshair(img: np.ndarray, x: int, y: int, size: int = 6, color=GREEN):
    """Draw [+] crosshair at (x, y)."""
    cv2.line(img, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x - 3, y - 3), (x + 3, y + 3), color, 1, cv2.LINE_AA)


def draw_dashed_line(img: np.ndarray, pt1, pt2, color, thickness: int = 1,
                     dash: int = 6, gap: int = 4):
    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    dist = float(np.linalg.norm(p2 - p1))
    if dist < 1:
        return
    seg = dash + gap
    n = int(dist / seg) + 1
    for i in range(n):
        t0 = (i * seg) / dist
        t1 = min(1.0, (i * seg + dash) / dist)
        a = (p1 + (p2 - p1) * t0).astype(int)
        b = (p1 + (p2 - p1) * t1).astype(int)
        cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)


def draw_skeleton_code(frame: np.ndarray, hand: HandData):
    """Draw hand skeleton using [+] crosshairs and dashed connections."""
    lm = hand.landmarks_px
    tips_set = set(FINGERTIP_IDS)

    for conn in HAND_CONNECTIONS:
        draw_dashed_line(frame, tuple(lm[conn[0]]), tuple(lm[conn[1]]),
                         DIM_GREEN, thickness=1, dash=4, gap=4)

    for i, pt in enumerate(lm):
        col = CYAN if i in tips_set else GREEN
        sz  = 6 if i in tips_set else 4
        draw_crosshair(frame, int(pt[0]), int(pt[1]), size=sz, color=col)


def pil_rotated_texts(frame_bgr: np.ndarray, items: list) -> np.ndarray:
    """
    Composite rotated text items onto frame_bgr using PIL (one pass).
    items: [(text, x, y, angle_deg, color_bgr, alpha_0_1), ...]
    """
    if not items:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    font = _load_mono_font(12)

    for text, x, y, angle, color_bgr, alpha in items:
        if alpha <= 0:
            continue
        b_c, g_c, r_c = int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])
        a_val = int(np.clip(alpha * 255, 0, 255))

        try:
            bbox = font.getbbox(text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = max(1, len(text) * 7), 14

        pad = 3
        surf = Image.new('RGBA', (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
        ImageDraw.Draw(surf).text((pad, pad), text, font=font,
                                  fill=(r_c, g_c, b_c, a_val))
        rotated = surf.rotate(angle, expand=True, resample=Image.BILINEAR)
        px = x - rotated.width // 2
        py = y - rotated.height // 2
        if -rotated.width < px < w and -rotated.height < py < h:
            overlay.paste(rotated, (px, py), rotated)

    base_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert('RGBA')
    composited = Image.alpha_composite(base_pil, overlay)
    return cv2.cvtColor(np.array(composited.convert('RGB')), cv2.COLOR_RGB2BGR)


# ── Effect base ───────────────────────────────────────────────────────────────

class Effect:
    name = "base"

    def update(self, hands: list, frame_w: int, frame_h: int):
        pass

    def draw(self, base: np.ndarray) -> np.ndarray:
        return base

    def clear(self):
        pass


# ── 1. MatrixRainEffect ───────────────────────────────────────────────────────

class MatrixRainEffect(Effect):
    name = "matrix_rain"
    COL_W = 10
    ROW_H = 14

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.n_cols  = w // self.COL_W
        self.n_rows  = h // self.ROW_H + 4

        self.head_y    = np.random.uniform(-h, 0, self.n_cols).astype(np.float32)
        self.base_spd  = np.random.uniform(2, 7, self.n_cols).astype(np.float32)
        self.speed     = self.base_spd.copy()
        self.trail_len = np.random.randint(8, 22, self.n_cols)
        self.chars     = np.random.randint(0, len(MATRIX_CHARS), (self.n_rows, self.n_cols))
        self._tick     = 0
        self._hands: list = []

    def update(self, hands, frame_w, frame_h):
        self._hands = hands

        # Speed boost: columns near any hand landmark move faster
        self.speed = self.base_spd.copy()
        col_idx = np.arange(self.n_cols, dtype=np.float32)
        for hand in hands:
            for lm in hand.landmarks_px:
                c = lm[0] / self.COL_W
                boost = np.maximum(0.0, 9.0 - np.abs(col_idx - c))
                self.speed += boost

        self.head_y += self.speed

        # Wrap columns that scroll off the bottom
        max_y = self.h + self.trail_len * self.ROW_H
        reset = self.head_y > max_y
        if reset.any():
            n = int(reset.sum())
            self.head_y[reset]    = -np.random.uniform(0, self.h * 0.5, n)
            self.base_spd[reset]  = np.random.uniform(2, 7, n)
            self.speed[reset]     = self.base_spd[reset]
            self.trail_len[reset] = np.random.randint(8, 22, n)

        # Randomly mutate a small fraction of characters each frame
        self._tick += 1
        if self._tick % 4 == 0:
            mask = np.random.random((self.n_rows, self.n_cols)) < 0.08
            self.chars[mask] = np.random.randint(0, len(MATRIX_CHARS), int(mask.sum()))

    def draw(self, base: np.ndarray) -> np.ndarray:
        layer = np.zeros_like(base)
        h, w = base.shape[:2]
        n_chars = len(MATRIX_CHARS)

        # Collect fingertip columns for extra-bright treatment
        tip_cols: set = set()
        for hand in self._hands:
            for tip_idx in FINGERTIP_IDS:
                tip_cols.add(int(hand.landmarks_px[tip_idx][0]) // self.COL_W)

        for col in range(self.n_cols):
            x = col * self.COL_W
            head_row = int(self.head_y[col]) // self.ROW_H
            trail = int(self.trail_len[col])
            near = col in tip_cols

            for t in range(trail):
                row = head_row - t
                if row < 0 or row >= self.n_rows:
                    continue
                y = row * self.ROW_H
                if y < 0 or y >= h:
                    continue

                char = MATRIX_CHARS[int(self.chars[row, col]) % n_chars]

                if t == 0:
                    color = BRIGHT_G
                    scale = 0.9
                else:
                    fade = (1.0 - t / trail) ** 1.3
                    g = int((255 if near else 200) * fade)
                    color = (0, g, int(g * 0.25))
                    scale = 0.8

                cv2.putText(layer, char, (x, y + self.ROW_H - 2),
                            cv2.FONT_HERSHEY_PLAIN, scale, color, 1, cv2.LINE_AA)

        layer = glow_blur(layer, 5)
        return additive_blend(base, layer)


# ── 2. CodeTrailEffect ────────────────────────────────────────────────────────

@dataclass
class _TrailItem:
    text:      str
    x:         float
    y:         float
    vx:        float
    vy:        float
    birth:     float
    color:     tuple
    angle:     float
    rot_speed: float


class CodeTrailEffect(Effect):
    name = "code_trail"
    MAX_ITEMS = 60
    LIFETIME  = 1.5

    def __init__(self):
        self._items: deque = deque(maxlen=self.MAX_ITEMS)
        self._color_idx = 0
        self._prev_tips: dict = {}

    def update(self, hands, frame_w, frame_h):
        now = time.time()
        for i, hand in enumerate(hands):
            tip = hand.landmarks_px[INDEX_TIP].astype(np.float32)
            prev = self._prev_tips.get(i)
            if prev is not None:
                vel = tip - prev
                speed = float(np.linalg.norm(vel))
                if speed > 3.0:
                    color = SYNTAX_COLORS[self._color_idx % len(SYNTAX_COLORS)]
                    self._color_idx += 1
                    self._items.append(_TrailItem(
                        text=random.choice(CODE_TOKENS),
                        x=float(tip[0]), y=float(tip[1]),
                        vx=vel[0] * 0.3 + random.gauss(0, 1.0),
                        vy=vel[1] * 0.3 + random.gauss(0, 1.5),
                        birth=now,
                        color=color,
                        angle=random.uniform(-30, 30),
                        rot_speed=random.uniform(-6, 6),
                    ))
            self._prev_tips[i] = tip.copy()

        # Prune stale tip refs
        for k in list(self._prev_tips.keys()):
            if k >= len(hands):
                del self._prev_tips[k]

        # Physics update (small fixed list — no vectorization overhead needed)
        for item in self._items:
            item.x += item.vx
            item.y += item.vy
            item.vy += 0.08
            item.vx *= 0.97
            item.vy *= 0.97
            item.angle += item.rot_speed

    def draw(self, base: np.ndarray) -> np.ndarray:
        now = time.time()
        render = [
            (it.text, int(it.x), int(it.y), it.angle, it.color,
             1.0 - (now - it.birth) / self.LIFETIME)
            for it in self._items
            if (now - it.birth) < self.LIFETIME
        ]
        return pil_rotated_texts(base, render)

    def clear(self):
        self._items.clear()
        self._prev_tips.clear()


# ── 3. SyntaxExplosionEffect ──────────────────────────────────────────────────

class SyntaxExplosionEffect(Effect):
    name = "syntax_explosion"
    MAX_TOKENS = 80
    LIFETIME   = 2.2

    def __init__(self):
        # Each token: [x, y, vx, vy, birth, token_str, color_tuple]
        self._tokens: list = []
        self._prev_fist: dict = {}
        self._cooldown: dict = {}

    def _is_fist(self, h: HandData) -> bool:
        return sum(h.fingers_up) <= 1

    def _is_open(self, h: HandData) -> bool:
        return sum(h.fingers_up) >= 4

    def has_content(self) -> bool:
        return len(self._tokens) > 0

    def update(self, hands, frame_w, frame_h):
        now = time.time()
        for i, hand in enumerate(hands):
            was_fist = self._prev_fist.get(i, False)
            cd = self._cooldown.get(i, 0)

            if was_fist and self._is_open(hand) and cd == 0:
                cx, cy = int(hand.palm_center[0]), int(hand.palm_center[1])
                n_new = min(40, self.MAX_TOKENS - len(self._tokens))
                for _ in range(n_new):
                    ang   = random.uniform(0, 2 * np.pi)
                    spd   = random.uniform(3, 13)
                    color = random.choice(SYNTAX_COLORS)
                    self._tokens.append([
                        float(cx), float(cy),
                        np.cos(ang) * spd, np.sin(ang) * spd,
                        now, random.choice(EXPLOSION_TOKENS), color,
                    ])
                self._cooldown[i] = 45

            self._prev_fist[i] = self._is_fist(hand)
            if cd > 0:
                self._cooldown[i] = cd - 1

        # Physics + expiry
        alive = []
        for t in self._tokens:
            if now - t[4] < self.LIFETIME:
                t[0] += t[2]; t[1] += t[3]
                t[3] += 0.18
                t[2] *= 0.96; t[3] *= 0.96
                alive.append(t)
        self._tokens = alive[:self.MAX_TOKENS]

    def draw(self, base: np.ndarray) -> np.ndarray:
        if not self._tokens:
            return base
        now = time.time()
        layer = np.zeros_like(base)
        hh, ww = base.shape[:2]
        for t in self._tokens:
            age   = now - t[4]
            alpha = max(0.0, 1.0 - age / self.LIFETIME)
            x, y  = int(t[0]), int(t[1])
            if not (0 <= x < ww and 0 <= y < hh):
                continue
            col = tuple(int(c * alpha) for c in t[6])
            cv2.putText(layer, t[5], (x, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, col, 1, cv2.LINE_AA)
        layer = glow_blur(layer, 7)
        return additive_blend(base, layer)

    def clear(self):
        self._tokens.clear()


# ── 4. BinaryPinchDrawEffect ──────────────────────────────────────────────────

class BinaryPinchDrawEffect(Effect):
    name = "binary_pinch_draw"

    def __init__(self):
        self._paths: list = []           # completed paths
        self._active_path: list = []     # path being drawn
        self._phase = 0.0
        self._pinch_state: dict = {}

    def update(self, hands, frame_w, frame_h):
        self._phase = (self._phase + 0.5) % 2.0

        for i, hand in enumerate(hands):
            pinching = hand.pinch_dist < 45
            was      = self._pinch_state.get(i, False)

            if pinching:
                mid = ((hand.landmarks_px[THUMB_TIP].astype(np.float32) +
                        hand.landmarks_px[INDEX_TIP].astype(np.float32)) / 2).astype(np.int32)
                pt = (int(mid[0]), int(mid[1]))
                if not was:
                    self._active_path = [pt]
                elif self._active_path:
                    last = np.array(self._active_path[-1])
                    if np.linalg.norm(np.array(pt) - last) > 4:
                        self._active_path.append(pt)
            else:
                if was and len(self._active_path) > 1:
                    self._paths.append(list(self._active_path))
                    self._active_path = []

            self._pinch_state[i] = pinching

    def _draw_binary_seg(self, layer, p1, p2, offset: float):
        p1a = np.array(p1, dtype=np.float32)
        p2a = np.array(p2, dtype=np.float32)
        seg  = float(np.linalg.norm(p2a - p1a))
        if seg < 1:
            return
        spacing = 8
        n = max(1, int(seg / spacing))
        for j in range(n):
            pos = p1a + (p2a - p1a) * (j / n)
            digit = int(j + offset) % 2
            color = GREEN if digit == 0 else CYAN
            cv2.putText(layer, "01"[digit], (int(pos[0]), int(pos[1])),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, color, 1, cv2.LINE_AA)

    def _draw_path(self, layer, path):
        for i in range(len(path) - 1):
            self._draw_binary_seg(layer, path[i], path[i + 1], self._phase)

    def draw(self, base: np.ndarray) -> np.ndarray:
        layer = np.zeros_like(base)
        for path in self._paths:
            self._draw_path(layer, path)
        if len(self._active_path) > 1:
            self._draw_path(layer, self._active_path)
        layer = glow_blur(layer, 7)
        return additive_blend(base, layer)

    def clear(self):
        self._paths.clear()
        self._active_path.clear()
        self._pinch_state.clear()


# ── 5. TerminalSummonEffect ───────────────────────────────────────────────────

class TerminalSummonEffect(Effect):
    name = "terminal_summon"

    CHARS_PER_FRAME = 2
    HOLD_FRAMES     = 55

    def __init__(self):
        self._active         = False
        self._center         = (320, 240)
        self._cmd_idx        = 0
        self._char_idx       = 0
        self._hold           = 0
        self._done_lines: list = []
        self._tick           = 0

    def _is_open(self, h: HandData) -> bool:
        return sum(h.fingers_up) >= 4

    def update(self, hands, frame_w, frame_h):
        self._tick += 1
        active_hand = next((h for h in hands if self._is_open(h)), None)
        self._active = active_hand is not None
        if not self._active:
            return

        self._center = (int(active_hand.palm_center[0]),
                        int(active_hand.palm_center[1]))

        cmd = COMMANDS[self._cmd_idx % len(COMMANDS)]
        if self._char_idx < len(cmd):
            self._char_idx = min(self._char_idx + self.CHARS_PER_FRAME, len(cmd))
            self._hold = 0
        else:
            self._hold += 1
            if self._hold >= self.HOLD_FRAMES:
                self._done_lines.append(f"$ {cmd}")
                if len(self._done_lines) > 4:
                    self._done_lines.pop(0)
                self._cmd_idx += 1
                self._char_idx = 0
                self._hold     = 0

    def draw(self, base: np.ndarray) -> np.ndarray:
        if not self._active:
            return base
        cx, cy = self._center
        hh, ww = base.shape[:2]

        win_w, win_h = 400, 140
        x0 = int(np.clip(cx - win_w // 2, 0, ww - win_w))
        y0 = int(np.clip(cy - win_h // 2, 0, hh - win_h))
        x1, y1 = x0 + win_w, y0 + win_h

        result = base.copy()

        # Translucent dark background
        bg = result.copy()
        cv2.rectangle(bg, (x0, y0), (x1, y1), (4, 12, 4), -1)
        cv2.addWeighted(bg, 0.85, result, 0.15, 0, result)

        # Outer border + title bar
        cv2.rectangle(result, (x0, y0), (x1, y1), GREEN, 1, cv2.LINE_AA)
        cv2.rectangle(result, (x0, y0), (x1, y0 + 18), GREEN, -1)
        cv2.putText(result, "  terminal_magic  v1.0 ", (x0 + 4, y0 + 13),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, BLACK, 1, cv2.LINE_AA)

        font  = cv2.FONT_HERSHEY_PLAIN
        lh    = 19
        ty    = y0 + 24

        for line in self._done_lines[-3:]:
            cv2.putText(result, line, (x0 + 6, ty), font, 0.85, DIM_GREEN, 1, cv2.LINE_AA)
            ty += lh

        cmd    = COMMANDS[self._cmd_idx % len(COMMANDS)]
        cur    = f"$ {cmd[:self._char_idx]}"
        blink  = "█" if (self._tick // 15) % 2 else " "
        cv2.putText(result, cur + blink, (x0 + 6, ty), font, 0.85, GREEN, 1, cv2.LINE_AA)

        return result


# ── 6. StackTraceCurseEffect ──────────────────────────────────────────────────

class StackTraceCurseEffect(Effect):
    name = "stack_trace_curse"

    STILL_PX      = 14
    STILL_NEEDED  = 28   # ~1 s at 30 fps

    def __init__(self):
        self._active       = False
        self._still        = 0
        self._tip_pos      = (0, 0)
        self._prev_tip: Optional[np.ndarray] = None
        self._scroll       = 0.0
        self._charge_shown = False

    def _is_point(self, h: HandData) -> bool:
        f = h.fingers_up
        return f[1] and not f[2] and not f[3] and not f[4]

    def update(self, hands, frame_w, frame_h):
        pointing = next((h for h in hands if self._is_point(h)), None)

        if pointing is None:
            self._still  = 0
            self._active = False
            self._prev_tip = None
            return

        tip = pointing.landmarks_px[INDEX_TIP].astype(np.float32)
        self._tip_pos = (int(tip[0]), int(tip[1]))

        if self._prev_tip is not None:
            moved = float(np.linalg.norm(tip - self._prev_tip))
            self._still = self._still + 1 if moved < self.STILL_PX else max(0, self._still - 2)
        self._prev_tip = tip.copy()

        if self._still >= self.STILL_NEEDED:
            self._active = True
            self._scroll = (self._scroll + 0.3) % len(STACK_TRACE)
        else:
            self._active = False

    def draw(self, base: np.ndarray) -> np.ndarray:
        tx, ty = self._tip_pos
        result = base.copy()

        # Charge bar while building up stillness
        if not self._active and self._still > 5:
            charge  = self._still / self.STILL_NEEDED
            bar_w   = int(60 * charge)
            cv2.rectangle(result, (tx - 30, ty - 22), (tx - 30 + bar_w, ty - 15), AMBER, -1)
            cv2.rectangle(result, (tx - 30, ty - 22), (tx + 30,         ty - 15), AMBER, 1)
            label = f"{int(charge*100)}%"
            cv2.putText(result, label, (tx - 10, ty - 24),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, AMBER, 1, cv2.LINE_AA)
            return result

        if not self._active:
            return result

        h, w = result.shape[:2]
        start = int(self._scroll)
        lh    = 15

        for i in range(min(11, len(STACK_TRACE))):
            line_idx = (start + i) % len(STACK_TRACE)
            line = STACK_TRACE[line_idx]
            ly   = ty + i * lh
            if ly >= h:
                break

            fade = min(1.0, (i + 1) / 4.0)
            if "Error" in line or "fault" in line or "exit" in line:
                base_col = RED
            elif "File" in line or "line" in line:
                base_col = AMBER
            elif "assert" in line.lower() or "raise" in line:
                base_col = PINK
            else:
                base_col = DIM_GREEN

            col = tuple(int(c * fade) for c in base_col)
            cv2.putText(result, line, (tx, ly),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, col, 1, cv2.LINE_AA)

        return result


# ── 7. TwoHandCompileEffect ───────────────────────────────────────────────────

class TwoHandCompileEffect(Effect):
    name = "two_hand_compile"

    def __init__(self):
        self._active     = False
        self._progress   = 0.0
        self._result: Optional[str] = None
        self._res_timer  = 0
        self._positions: list = []

    def _both_open(self, hands) -> bool:
        return len(hands) >= 2 and all(sum(h.fingers_up) >= 4 for h in hands)

    def update(self, hands, frame_w, frame_h):
        self._active    = self._both_open(hands)
        self._positions = [h.palm_center.astype(np.int32)
                           for h in hands[:2]] if len(hands) >= 2 else []

        if self._res_timer > 0:
            self._res_timer -= 1
            if self._res_timer == 0:
                self._result   = None
                self._progress = 0.0
            return

        if self._active:
            self._progress = min(1.0, self._progress + 0.007)
            if self._progress >= 1.0:
                self._result    = "success" if random.random() < 0.6 else "fault"
                self._res_timer = 100
        else:
            self._progress = max(0.0, self._progress - 0.025)

    def draw(self, base: np.ndarray) -> np.ndarray:
        if not self._active and self._progress <= 0 and self._result is None:
            return base

        result = base.copy()
        hh, ww = result.shape[:2]

        # Flash result message
        if self._result is not None:
            fade = min(1.0, self._res_timer / 35.0)
            if self._result == "success":
                msg   = "BUILD SUCCESS  [OK]"
                color = tuple(int(c * fade) for c in GREEN)
            else:
                msg   = "SEGMENTATION FAULT"
                color = tuple(int(c * fade) for c in RED)

            ts = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)[0]
            tx = (ww - ts[0]) // 2
            ty = hh // 2

            glow = np.zeros_like(result)
            cv2.putText(glow, msg, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3, cv2.LINE_AA)
            glow = glow_blur(glow, 23)
            result = additive_blend(result, glow)
            cv2.putText(result, msg, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2, cv2.LINE_AA)
            return result

        if len(self._positions) < 2:
            return result

        p1, p2 = self._positions[0], self._positions[1]
        mid    = ((p1 + p2) / 2).astype(np.int32)
        bar_w  = min(320, int(np.linalg.norm(p2 - p1)) + 60)
        bar_h  = 22
        bx     = mid[0] - bar_w // 2
        by     = mid[1] - bar_h // 2

        cv2.rectangle(result, (bx, by), (bx + bar_w, by + bar_h), DIM_GREEN, 1)
        fill = int(bar_w * self._progress)
        if fill > 0:
            cv2.rectangle(result, (bx, by), (bx + fill, by + bar_h), GREEN, -1)

        pct   = int(self._progress * 100)
        label = f"COMPILING... {pct}%"
        cv2.putText(result, label, (bx, by - 8),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, AMBER, 1, cv2.LINE_AA)

        # Lines from hands to bar ends
        cv2.line(result, tuple(p1), (bx, mid[1]),          DIM_GREEN, 1, cv2.LINE_AA)
        cv2.line(result, tuple(p2), (bx + bar_w, mid[1]),  DIM_GREEN, 1, cv2.LINE_AA)

        return result


# ── 8. HUDEffect (always-on) ──────────────────────────────────────────────────

class HUDEffect:
    """Terminal-style HUD — always drawn last."""

    def __init__(self):
        self._gesture     = "none"
        self._effect_name = "matrix_rain"
        self._fps         = 0.0
        self._mode        = "auto"
        self._coords      = (0.0, 0.0)
        self._tick        = 0

    def set_info(self, gesture: str, effect_name: str, fps: float,
                 mode: str, coords=(0.0, 0.0)):
        self._gesture     = gesture
        self._effect_name = effect_name
        self._fps         = fps
        self._mode        = mode
        self._coords      = coords
        self._tick       += 1

    def draw(self, base: np.ndarray) -> np.ndarray:
        hh, ww = base.shape[:2]
        result = base.copy()

        pw, ph = 268, 114
        bg = result.copy()
        cv2.rectangle(bg, (5, 5), (5 + pw, 5 + ph), (0, 0, 0), -1)
        cv2.addWeighted(bg, 0.65, result, 0.35, 0, result)
        cv2.rectangle(result, (5, 5), (5 + pw, 5 + ph), DIM_GREEN, 1)

        font = cv2.FONT_HERSHEY_PLAIN
        blink = "█" if (self._tick // 20) % 2 else " "
        lines = [
            (f"fps: {self._fps:.0f}",                             GREEN),
            (f"// {self._effect_name}",                           CYAN),
            (f"gesture: {self._gesture}",                         AMBER),
            (f"x: {self._coords[0]:.2f}  y: {self._coords[1]:.2f}", DIM_GREEN),
            (f"mode: {self._mode}  {blink}",                      GREEN),
        ]

        ty = 22
        for text, color in lines:
            cv2.putText(result, text, (11, ty), font, 0.85, color, 1, cv2.LINE_AA)
            ty += 18

        return result


# ── Gesture classifier ────────────────────────────────────────────────────────

def detect_gesture(hands: list) -> str:
    if not hands:
        return "none"

    if len(hands) >= 2:
        return "dual_palm" if all(sum(h.fingers_up) >= 4 for h in hands) else "two_hands"

    hand = hands[0]
    f    = hand.fingers_up
    n    = sum(f)

    if hand.pinch_dist < 45:
        return "pinch"
    if n == 0:
        return "fist"
    if n >= 4:
        return "open_palm"
    if f[1] and not f[2] and not f[3] and not f[4]:
        return "point"
    if f[1] and f[2] and not f[3] and not f[4]:
        return "peace"
    return "custom"
