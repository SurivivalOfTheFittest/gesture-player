#!/usr/bin/env python3
"""Two-hand gesture synth: left hand picks the chord, right hand shapes it."""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError:
    print("Missing packages. Run:  ./run.sh", file=sys.stderr)
    raise SystemExit(1)

from src.clip import Clip
from src.devices import (
    LiveCam, find_camera, find_loopback, find_microphone, find_speakers,
    list_devices, open_camera, sample_rate,
)
from src.hands import Hands
from src.synth import Synth

BG_HEX = "#2D1C2C"


def _photo(bgr: np.ndarray) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


def _key(event: tk.Event) -> int:
    special = {
        "Return": 13, "KP_Enter": 13, "Escape": 27, "Tab": 9, "space": 32,
        "Right": 63235, "BackSpace": 8, "Delete": 8,
    }
    if event.keysym in special:
        return special[event.keysym]
    if event.char:
        return ord(event.char)
    return -1


def main() -> int:
    p = argparse.ArgumentParser(description="Throwing Hands (WIP)")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()
    if args.list_devices:
        list_devices()
        return 0

    speakers, camera = find_speakers(args.output_device), find_camera(args.camera)
    tap, mic = find_loopback(), find_microphone()
    sr = sample_rate(speakers)
    print(f"Output [{speakers}]  Camera [{camera}]  {sr} Hz")
    print("Click a circle, or press space / right arrow for the next. Enter hides empty. Q = quit.")
    print("REC in the left panel records the window, chords, and singing into recordings/.")
    if tap is not None:
        print("Cmd+Shift+5 still works: Options → Microphone → BlackHole 2ch.\n")
    else:
        print("Cmd+Shift+5 is silent until BlackHole is installed; use the REC button instead.\n")

    try:
        hands = Hands()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    cap = open_camera(camera)
    if not cap.isOpened():
        hands.close()
        print("Could not open camera. Allow Camera access, then retry.", file=sys.stderr)
        return 1
    live = LiveCam(cap)

    synth = None
    keys: list[int] = []
    running = True
    clip = Clip()
    last_size = (1280, 860)

    root = tk.Tk()
    root.title("Throwing Hands (WIP)")
    root.configure(bg=BG_HEX, highlightthickness=0)
    root.geometry("1280x860")
    root.minsize(720, 500)
    view = tk.Canvas(root, bd=0, highlightthickness=0, bg=BG_HEX)
    view.pack(fill=tk.BOTH, expand=True)

    def on_close() -> None:
        nonlocal running
        running = False

    def finish_rec() -> None:
        chunks = synth.toggle_capture() if synth and synth.capturing else []
        path = clip.stop(chunks or [], sr)
        if path:
            print(f"Saved {path}")

    def toggle_rec() -> None:
        if clip.active:
            finish_rec()
            return
        if synth is None:
            return
        clip.start(last_size)
        synth.toggle_capture()
        print("Recording… click STOP when you are done.")

    def on_click(event: tk.Event) -> None:
        if hands.click(event.x, event.y) == "rec":
            toggle_rec()

    def on_key(event: tk.Event) -> None:
        keys.append(_key(event))

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Key>", on_key)
    view.bind("<Button-1>", on_click)
    view.focus_set()
    root.update_idletasks()

    try:
        synth = Synth(speakers, sr, tap=tap, mic=mic)
        synth.start()
        last_gen = -1
        while running:
            ok, frame, gen = live.read()
            if not ok or frame is None:
                time.sleep(0.004)
                root.update()
                continue
            w, h = max(view.winfo_width(), 1), max(view.winfo_height(), 1)
            if w < 200 or h < 200:
                w, h = 1280, 860
            last_size = (w, h)
            if gen != last_gen:
                last_gen = gen
                frame, g = hands.process(frame, size=(w, h), rec=clip.active)
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                synth.set_chord(g.freqs, g.volume if g.playing else 0.0, g.cutoff if g.playing else 0.4)
                if clip.active:
                    clip.write(frame)
                photo = _photo(frame)
                view.delete("img")
                view.create_image(0, 0, image=photo, anchor="nw", tags="img")
                view.image = photo
            else:
                time.sleep(0.002)
            root.update()
            while keys:
                action = hands.handle_key(keys.pop(0))
                if action == "quit":
                    running = False
                    break
    except tk.TclError:
        pass
    except Exception as exc:
        print(f"Audio failed: {exc}\nTry: python main.py --list-devices", file=sys.stderr)
        return 1
    finally:
        if clip.active:
            finish_rec()
        if synth:
            synth.stop()
        live.stop()
        try:
            root.destroy()
        except tk.TclError:
            pass
        hands.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nQuit.")
        raise SystemExit(0)
