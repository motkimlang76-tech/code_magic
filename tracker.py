"""Hand tracking wrapper — uses MediaPipe Tasks API (mediapipe >= 0.10)."""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field

from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

# ── Model download ────────────────────────────────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


def _ensure_model():
    if _MODEL_PATH.exists():
        return
    print(f"Downloading hand_landmarker.task (~8 MB) ...", flush=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Model ready.", flush=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not download hand_landmarker.task: {e}\n"
            f"Download manually from:\n  {_MODEL_URL}\n"
            f"and place it next to tracker.py"
        ) from e


# ── Landmark index constants ──────────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP   = 1, 2, 3, 4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

FINGERTIP_IDS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIP_IDS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

# Hardcoded connections (identical to the old mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = frozenset([
    (0,  1), (1,  2), (2,  3), (3,  4),
    (0,  5), (5,  6), (6,  7), (7,  8),
    (0,  9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5,  9), (9, 13), (13, 17),
])


@dataclass
class HandData:
    landmarks_norm: np.ndarray   # (21, 2) normalized [0,1]
    landmarks_px:   np.ndarray   # (21, 2) pixel coords
    handedness:     str          # 'Left' or 'Right'
    fingers_up:     list         # [thumb, index, middle, ring, pinky]
    pinch_dist:     float        # pixels, thumb tip ↔ index tip
    palm_center:    np.ndarray   # (2,) pixel
    velocity:       np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )


class HandTracker:
    def __init__(self, max_hands: int = 2,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        _ensure_model()

        options = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(
                model_asset_path=str(_MODEL_PATH)
            ),
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = _mp_vision.HandLandmarker.create_from_options(options)
        self._smooth: dict[int, np.ndarray] = {}
        self._prev_palms: dict[int, np.ndarray] = {}
        # Monotonic timestamps for VIDEO mode (must strictly increase)
        self._t0_ms = int(time.monotonic() * 1000)

    def process(self, frame_rgb: np.ndarray) -> list[HandData]:
        h, w = frame_rgb.shape[:2]

        ts_ms = int(time.monotonic() * 1000) - self._t0_ms + 1
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_img, ts_ms)

        out: list[HandData] = []
        if not result.hand_landmarks:
            self._smooth.clear()
            self._prev_palms.clear()
            return out

        for idx, (lm_list, hdness) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            label = hdness[0].category_name  # 'Left' or 'Right'

            raw = np.array([[lm.x, lm.y] for lm in lm_list], dtype=np.float32)

            # EMA smoothing (α = 0.5)
            if idx in self._smooth:
                raw = 0.5 * raw + 0.5 * self._smooth[idx]
            self._smooth[idx] = raw.copy()

            lm_px = (raw * np.array([w, h])).astype(np.int32)

            fingers = self._finger_states(raw, label)
            pinch   = float(np.linalg.norm(lm_px[THUMB_TIP] - lm_px[INDEX_TIP]))

            mcp_ids = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
            palm    = lm_px[mcp_ids].mean(axis=0).astype(np.float32)
            vel     = (
                palm - self._prev_palms[idx]
                if idx in self._prev_palms
                else np.zeros(2, dtype=np.float32)
            )
            self._prev_palms[idx] = palm.copy()

            out.append(HandData(
                landmarks_norm=raw, landmarks_px=lm_px,
                handedness=label, fingers_up=fingers,
                pinch_dist=pinch, palm_center=palm, velocity=vel,
            ))

        for k in list(self._smooth.keys()):
            if k >= len(out):
                self._smooth.pop(k, None)
                self._prev_palms.pop(k, None)

        return out

    def _finger_states(self, norm: np.ndarray, handedness: str) -> list[bool]:
        up = []
        if handedness == 'Right':
            up.append(bool(norm[THUMB_TIP][0] < norm[THUMB_IP][0]))
        else:
            up.append(bool(norm[THUMB_TIP][0] > norm[THUMB_IP][0]))
        for tip, pip in zip(FINGERTIP_IDS[1:], FINGER_PIP_IDS[1:]):
            up.append(bool(norm[tip][1] < norm[pip][1]))
        return up

    def close(self):
        self._landmarker.close()
