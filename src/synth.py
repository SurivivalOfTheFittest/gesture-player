"""Chord synth: sines at equal-tempered frequencies, mix voice into the screen-record tap."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import sounddevice as sd


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


class _Ring:
    def __init__(self, n: int) -> None:
        self.buf = np.zeros(n, np.float32)
        self.n = n
        self.w = 0
        self.avail = 0
        self.lock = threading.Lock()

    def push(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, np.float32).ravel()
        k = min(x.size, self.n)
        x = x[-k:]
        with self.lock:
            i, n = self.w, self.n
            if i + k <= n:
                self.buf[i : i + k] = x
            else:
                a = n - i
                self.buf[i:] = x[:a]
                self.buf[: k - a] = x[a:]
            self.w = (i + k) % n
            self.avail = min(n, self.avail + k)

    def pop(self, k: int) -> np.ndarray:
        out = np.zeros(k, np.float32)
        with self.lock:
            take = min(k, self.avail)
            if take <= 0:
                return out
            start = (self.w - self.avail) % self.n
            n = self.n
            if start + take <= n:
                out[:take] = self.buf[start : start + take]
            else:
                a = n - start
                out[:a] = self.buf[start:]
                out[a:take] = self.buf[: take - a]
            self.avail -= take
        return out


class Synth:
    def __init__(
        self,
        speakers: int,
        sample_rate: int,
        volume: float = 0.22,
        tap: int | None = None,
        mic: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.volume = volume
        self._lock = threading.Lock()
        self._freqs = np.zeros(4)
        self._phase = np.zeros(4)
        self._amp = 0.0
        self._target = 0.0
        self._cutoff = 0.4
        self._mic_ring = _Ring(sample_rate)
        self._mix_q: deque[np.ndarray] = deque(maxlen=12)
        self._last_mix = np.zeros(2048, np.float32)
        self.capturing = False
        self.takes: list[np.ndarray] = []
        kw = dict(samplerate=sample_rate, blocksize=2048, dtype="float32", latency=0.08)
        self._stream = sd.OutputStream(device=speakers, channels=2, callback=self._callback, **kw)
        self._tap = None
        self._mic = None
        if tap is not None and tap != speakers:
            self._tap = sd.OutputStream(device=tap, channels=2, callback=self._tap_callback, **kw)
        if mic is not None:
            self._mic = sd.InputStream(device=mic, channels=1, callback=self._mic_callback, **kw)

    def start(self) -> None:
        if self._mic:
            try:
                self._mic.start()
            except Exception as exc:
                print(f"Could not open microphone ({exc}). Screen recordings will have chords only.")
                self._mic = None
        self._stream.start()
        if self._tap:
            self._tap.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()
        if self._tap:
            self._tap.stop()
            self._tap.close()
        if self._mic:
            self._mic.stop()
            self._mic.close()

    def toggle_capture(self) -> list[np.ndarray] | None:
        with self._lock:
            if self.capturing:
                self.capturing = False
                chunks, self.takes = self.takes, []
                return chunks
            self.capturing, self.takes = True, []
            return None

    def set_chord(self, freqs: list[float], amp: float, cutoff: float) -> None:
        with self._lock:
            self._freqs[:] = (list(freqs) + [0, 0, 0, 0])[:4]
            self._target = float(np.clip(amp, 0, 1))
            self._cutoff = float(np.clip(cutoff, 0.08, 0.9))

    def _mic_callback(self, indata, _frames, _time, _status) -> None:
        self._mic_ring.push(indata[:, 0])

    def _callback(self, outdata, frames, _time, _status) -> None:
        with self._lock:
            freqs, target, cutoff = self._freqs.copy(), self._target, self._cutoff

        mix = np.zeros(frames, np.float64)
        t = np.arange(frames, dtype=np.float64)
        n = 0
        for i, freq in enumerate(freqs):
            if freq < 20:
                continue
            step = 2 * np.pi * freq / self.sample_rate
            phase = self._phase[i] + step * t
            mix += np.sin(phase)
            self._phase[i] = (self._phase[i] + step * frames) % (2 * np.pi)
            n += 1
        if n:
            mix /= n

        amp_end = self._amp + 0.15 * (target - self._amp)
        mix *= np.linspace(self._amp, amp_end, frames, dtype=np.float64) * self.volume
        self._amp = float(amp_end)

        wet = 1.0 - float(cutoff)
        if wet > 0.02:
            taps = 5
            kernel = np.ones(taps) / taps
            padded = np.pad(mix, (taps - 1, 0), mode="edge")
            mix = mix * (1.0 - wet) + np.convolve(padded, kernel, mode="valid") * wet

        synth = np.clip(mix, -1, 1).astype(np.float32)
        outdata[:, 0] = synth
        outdata[:, 1] = synth
        voice = self._mic_ring.pop(frames)
        mixed = np.clip(synth + voice * 0.9, -1, 1).astype(np.float32)
        with self._lock:
            self._mix_q.append(mixed)
            self._last_mix = mixed
            if self.capturing:
                self.takes.append(mixed.copy())

    def _tap_callback(self, outdata, frames, _time, _status) -> None:
        with self._lock:
            mixed = self._mix_q.popleft() if self._mix_q else self._last_mix
        if mixed.size == frames:
            outdata[:, 0] = mixed
            outdata[:, 1] = mixed
        elif mixed.size > frames:
            outdata[:, 0] = mixed[:frames]
            outdata[:, 1] = mixed[:frames]
        else:
            outdata.fill(0)
            outdata[: mixed.size, 0] = mixed
            outdata[: mixed.size, 1] = mixed
