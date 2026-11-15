"""Laptop camera and speakers."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager

import cv2
import numpy as np
import sounddevice as sd


def list_devices() -> None:
    print(sd.query_devices())


_SKIP_OUT = ("iphone", "ipad", "blackhole", "loopback", "soundflower", "aggregate", "multi-output")


def find_speakers(index: int | None = None) -> int:
    if index is not None:
        return index
    devices = sd.query_devices()
    infos = [devices] if isinstance(devices, dict) else list(devices)
    macbook = fallback = None
    for i, info in enumerate(infos):
        if info["max_output_channels"] < 1:
            continue
        name = info["name"].lower()
        if any(tag in name for tag in _SKIP_OUT):
            continue
        if macbook is None and "macbook" in name and "speaker" in name:
            macbook = i
        if fallback is None:
            fallback = i
    if macbook is not None:
        return macbook
    if fallback is not None:
        return fallback
    raise RuntimeError("No speakers found. Pass --output-device.")


def find_microphone() -> int | None:
    devices = sd.query_devices()
    infos = [devices] if isinstance(devices, dict) else list(devices)
    macbook = fallback = None
    for i, info in enumerate(infos):
        if info["max_input_channels"] < 1:
            continue
        name = info["name"].lower()
        if any(tag in name for tag in ("iphone", "ipad", "blackhole", "loopback", "soundflower")):
            continue
        if macbook is None and "macbook" in name and "microphone" in name:
            macbook = i
        if fallback is None:
            fallback = i
    return macbook if macbook is not None else fallback


def find_loopback() -> int | None:
    """Virtual cable so Cmd+Shift+5 can record the synth (pick it as Microphone)."""
    devices = sd.query_devices()
    infos = [devices] if isinstance(devices, dict) else list(devices)
    for i, info in enumerate(infos):
        if info["max_output_channels"] < 1:
            continue
        if "blackhole" in info["name"].lower():
            return i
    return None


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


class LiveCam:
    """Keep only the newest camera frame so the UI never plays a stale buffer."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self._lock = threading.Lock()
        self._ok = False
        self._frame: np.ndarray | None = None
        self._gen = 0
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while self._run:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            with self._lock:
                self._ok, self._frame, self._gen = True, frame, self._gen + 1

    def read(self) -> tuple[bool, np.ndarray | None, int]:
        with self._lock:
            if self._frame is None:
                return False, None, self._gen
            return self._ok, self._frame, self._gen

    def stop(self) -> None:
        self._run = False
        self._t.join(timeout=0.5)
        self.cap.release()


@contextmanager
def _quiet():
    saved, null = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    os.close(null)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def find_camera(index: int | None = None) -> int:
    """FaceTime is usually last; Continuity Camera (iPhone) is first."""
    if index is not None:
        return index
    with _quiet():
        for i in range(7, -1, -1):
            cap = open_camera(i)
            ok = cap.isOpened()
            cap.release()
            if ok:
                return i
    return 0


def sample_rate(device: int) -> int:
    return int(sd.query_devices(device)["default_samplerate"])
