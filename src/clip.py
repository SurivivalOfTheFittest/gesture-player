"""Save a take of the app window plus mixed chords and voice."""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RECORDINGS = Path(__file__).resolve().parent.parent / "recordings"
FPS = 20
MAX_W, MAX_H = 1280, 720


def _ffmpeg() -> str | None:
    for name in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        path = shutil.which(name) if "/" not in name else name
        if path and Path(path).is_file():
            return path
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None


def _fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if frame.shape[1] == w and frame.shape[0] == h:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


class Clip:
    def __init__(self) -> None:
        self.active = False
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._size = (0, 0)
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._frames = 0
        self._t0 = 0.0
        self._run = False

    def start(self, size: tuple[int, int]) -> None:
        RECORDINGS.mkdir(exist_ok=True)
        w, h = max(2, size[0]), max(2, size[1])
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        self._size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
        self._q = queue.Queue(maxsize=2)
        self._path = RECORDINGS / f"throwing_hands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path.with_suffix(".tmp.mp4")), fourcc, FPS, self._size)
        self._frames = 0
        self._t0 = time.monotonic()
        self.active = True
        self._run = True
        self._thread = threading.Thread(target=self._encode, daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        if not self.active:
            return
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def _latest(self) -> np.ndarray | None:
        last = None
        while True:
            try:
                last = self._q.get_nowait()
            except queue.Empty:
                return last

    def _encode(self) -> None:
        period = 1.0 / FPS
        last: np.ndarray | None = None
        t_next = time.monotonic()
        while self._run or not self._q.empty():
            got = self._latest()
            if got is not None:
                last = got
            now = time.monotonic()
            if last is not None and now >= t_next and self._writer is not None:
                self._writer.write(_fit(last, self._size))
                self._frames += 1
                t_next += period
                if now - t_next > 0.25:
                    t_next = now
            else:
                time.sleep(0.004)

    def stop(self, chunks: list[np.ndarray], sr: int) -> Path | None:
        if not self.active:
            return None
        self.active = False
        self._run = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._writer:
            self._writer.release()
            self._writer = None
        raw = self._path.with_suffix(".tmp.mp4") if self._path else None
        wav = self._path.with_suffix(".wav") if self._path else None
        if raw is None or wav is None or self._path is None:
            return None
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        exe = _ffmpeg()
        if exe:
            for vcodec in ("libx264", "mpeg4"):
                cmd = [
                    exe, "-y", "-i", str(raw), "-i", str(wav),
                    "-c:v", vcodec, "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(self._path),
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    raw.unlink(missing_ok=True)
                    wav.unlink(missing_ok=True)
                    return self._path
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
        print(f"Could not mux audio into the movie. Video is {raw}, sound is {wav}.")
        return raw
