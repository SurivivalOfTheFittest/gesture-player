"""Track two hands and map them to the six chord squares."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import hand_landmarker

from src.synth import midi_to_hz
from src.ui import compose

MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
DEFAULT = ["Amaj7", "F#m7", "C#m7", "B", "", ""]
PC = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
    "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}
QUALITY = {
    "maj7": "maj7", "ma7": "maj7", "min7": "min7", "m7": "min7", "min": "minor",
    "m": "minor", "maj": "major", "7": "dom7", "dim": "dim", "sus4": "sus4", "sus2": "sus2",
}
STEPS = {
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "dom7": (0, 4, 7, 10),
    "major": (0, 4, 7), "minor": (0, 3, 7), "dim": (0, 3, 6),
    "sus4": (0, 5, 7), "sus2": (0, 2, 7),
}
BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
)


def parse_one(token: str) -> tuple[str, int, str] | None:
    raw = token.strip().replace("♯", "#").replace("♭", "b")
    if not raw:
        return None
    text = raw[0].upper() + raw[1:]
    root, rest = (text[:2], text[2:]) if len(text) > 1 and text[1] in "#b" else (text[0], text[1:])
    key = root[0].upper() + ("B" if len(root) == 2 and root[1] == "b" else "#" if len(root) == 2 else "")
    pc = PC.get(key)
    if pc is None:
        return None
    rest = rest.strip().lower()
    quality = "major"
    if rest:
        if rest not in QUALITY:
            return None
        quality = QUALITY[rest]
    midi = 48 + pc
    midi += 12 if midi < 54 else 0
    midi -= 12 if midi > 70 else 0
    return text, midi, quality


@dataclass
class Gesture:
    note: str = ""
    index: int = 0
    freqs: list[float] = field(default_factory=list)
    volume: float = 0.0
    cutoff: float = 0.4
    playing: bool = False


class Hands:
    def __init__(self) -> None:
        if not MODEL.is_file():
            raise FileNotFoundError(f"Missing {MODEL}. Run ./run.sh to download the hand model.")
        options = hand_landmarker.HandLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.55, min_tracking_confidence=0.5,
        )
        self._lm = hand_landmarker.HandLandmarker.create_from_options(options)
        self._ms = 0
        self.slots = list(DEFAULT)
        self.selected = 0
        self.editing = False
        self.locked = False
        self.extra: list[int] = []
        self.box_rects: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * 6
        self.title_rect = (0, 0, 0, 0)
        self.rec_rect = (0, 0, 0, 0)
        self._left = self._right = None
        self._pending = None
        self._dlock = threading.Lock()
        self._wake = threading.Event()
        self._run = True
        self._dt = threading.Thread(target=self._detect_loop, daemon=True)
        self._dt.start()

    def close(self) -> None:
        self._run = False
        self._wake.set()
        self._dt.join(timeout=0.5)
        self._lm.close()

    def visible(self) -> list[int]:
        if not self.locked:
            return list(range(6))
        filled = [i for i, s in enumerate(self.slots) if s.strip()]
        extra = [i for i in self.extra if i not in filled]
        shown = sorted(set(filled + extra))
        return shown or list(range(6))

    def click(self, x: int, y: int) -> str:
        rx1, ry1, rx2, ry2 = self.rec_rect
        if rx1 <= x <= rx2 and ry1 <= y <= ry2:
            return "rec"
        tx1, ty1, tx2, ty2 = self.title_rect
        if tx1 <= x <= tx2 and ty1 <= y <= ty2:
            self.locked, self.editing, self.extra = False, True, []
            return ""
        for i, (x1, y1, x2, y2) in enumerate(self.box_rects):
            if x2 <= x1:
                continue
            cx, cy, r = (x1 + x2) // 2, (y1 + y2) // 2, (x2 - x1) // 2
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                self.selected, self.editing = i, True
                return ""
        return ""

    def _open_next(self) -> int | None:
        shown = set(self.visible())
        for i in range(6):
            if i not in shown:
                self.extra.append(i)
                return i
        return None

    def _next(self) -> None:
        vis = self.visible()
        if self.selected in vis:
            i = vis.index(self.selected)
            if i + 1 < len(vis):
                self.selected = vis[i + 1]
            else:
                added = self._open_next()
                self.selected = added if added is not None else vis[0]
        else:
            self.selected = vis[0]
        self.editing = True

    def handle_key(self, key: int) -> str:
        if key in (ord("q"), ord("Q")):
            return "quit"
        if key == 27:
            if self.editing:
                self.editing = False
                return "edit"
            return "quit"
        if key in (32, 9, 3, 63235, 2555904, 65363):
            self._next()
            return "edit"
        if key in (13, 10):
            self.editing = False
            self.locked = True
            self.extra = []
            vis = self.visible()
            if vis and self.selected not in vis:
                self.selected = vis[0]
            return "edit"
        if key > 255 or not (self.editing or 32 < key < 127):
            return ""
        self.editing = True
        if key in (8, 127):
            self.slots[self.selected] = self.slots[self.selected][:-1]
        elif 32 < key < 127:
            self.slots[self.selected] += chr(key)
        return "edit"

    def _offer(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        small = frame if h <= 256 else cv2.resize(frame, (int(w * 256 / h), 256), interpolation=cv2.INTER_LINEAR)
        with self._dlock:
            self._pending = small
        self._wake.set()

    def _detect_loop(self) -> None:
        while self._run:
            self._wake.wait(timeout=0.05)
            self._wake.clear()
            if not self._run:
                break
            with self._dlock:
                small = self._pending
                self._pending = None
            if small is None:
                continue
            ms = int(time.monotonic() * 1000)
            if ms <= self._ms:
                ms = self._ms + 1
            self._ms = ms
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            result = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
            left = right = None
            if result.hand_landmarks:
                for i, pts in enumerate(result.hand_landmarks):
                    raw = result.handedness[i][0].category_name if result.handedness[i] else "Left"
                    if raw == "Right":
                        left = pts
                    else:
                        right = pts
            self._left, self._right = left, right

    def process(self, frame: np.ndarray, size: tuple[int, int] = (1280, 860), rec: bool = False) -> tuple[np.ndarray, Gesture]:
        self._offer(frame)
        shown = self.visible()
        g = _music(self._left, self._right, self.slots, shown)
        if self._left:
            _hand(frame, self._left)
        if self._right:
            _hand(frame, self._right)
        valid = [not s.strip() or parse_one(s) is not None for s in self.slots]
        canvas, self.box_rects, self.title_rect, self.rec_rect = compose(
            frame, g, self.slots, self.selected, self.editing, valid, shown, size, rec,
        )
        return canvas, g


def _flags(pts, hand: str) -> tuple[bool, bool, bool, bool, bool]:
    fingers = tuple(pts[t].y < pts[p].y - 0.02 for t, p in ((8, 6), (12, 10), (16, 14), (20, 18)))
    thumb = pts[4].x > pts[2].x + 0.025 if hand == "Left" else pts[4].x < pts[2].x - 0.025
    return (thumb,) + fingers


def _music(left, right, slots: list[str], shown: list[int]) -> Gesture:
    g = Gesture()
    if left is None or not shown:
        return g
    pinch = float(np.hypot(left[4].x - left[8].x, left[4].y - left[8].y)) < 0.055
    if pinch and len(shown) >= 6:
        slot = shown[5]
    else:
        n = min(sum(_flags(left, "Left")), 5, len(shown))
        if n <= 0:
            return g
        slot = shown[n - 1]
    parsed = parse_one(slots[slot])
    if parsed is None:
        return g
    name, root, quality = parsed
    octave, vol, cutoff = 0, 0.55, 0.4
    if right is not None:
        octave = 12 if _flags(right, "Right")[0] else 0
        vol = float(np.clip(1.15 - right[0].y, 0.12, 1.0))
        cutoff = float(np.clip(0.55 - (right[8].x - right[0].x), 0.12, 0.85))
    g.note, g.index, g.volume, g.cutoff, g.playing = name, slot + 1, vol, cutoff, True
    g.freqs = [midi_to_hz(root + octave + s) for s in STEPS[quality]]
    return g


def _hand(frame, pts) -> None:
    h, w = frame.shape[:2]
    xy = [(int(p.x * w), int(p.y * h)) for p in pts]
    for a, b in BONES:
        cv2.line(frame, xy[a], xy[b], (190, 190, 190), 1, cv2.LINE_AA)
    r = max(3, round(max(h, w) / 280))
    for p in xy:
        cv2.circle(frame, p, r, (240, 240, 240), -1, cv2.LINE_AA)
