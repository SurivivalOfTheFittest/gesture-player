"""Layout for the gesture-synth window. Colors from the sketch (#2D1C2C)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (0x2C, 0x1C, 0x2D)  # #2D1C2C
INK = (0xDE, 0xD6, 0xF0)  # #F0D6DE
MUTED = (0x96, 0x7A, 0xB4)  # #B47A96
HINT = (0xD2, 0xC6, 0xE4)  # #E4C6D2
STROKE = (0xB4, 0xA3, 0xC9)  # #C9A3B4
PLAY = (0xC8, 0xE8, 0xFF)  # #FFE8C8
FILL = (0x38, 0x24, 0x3A)  # #3A2438
HINTS = (
    "right hand",
    "up = louder",
    "down = quieter",
    "tilt = brighter / darker",
    "thumb out = octave up",
)
KEYS = ("Q = quit",)
FONT = Path(__file__).resolve().parent.parent / "fonts" / "Mahifont-Regular.ttf"
FALLBACK_FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
_MAHI = set(" !\"%&'+,-.0123456789:=?@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_FONTS: dict[tuple[int, str], ImageFont.FreeTypeFont] = {}
_PENDING: list[tuple] = []
_DECOR_KEY = None
_DECOR_IMG = None


def _hex(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    r, g, b = int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16)
    return (b, g, r)


GOLD_PLAY = [_hex(c) for c in ("#AE8625", "#F7EF8A", "#D2AC47", "#EDC967")]
GOLD_IDLE = [_hex(c) for c in ("#DFBD69", "#926F34")]
STAR = _hex("#F7EF8A")
STAR_DIM = _hex("#D2AC47")
SWIRL = _hex("#AE8625")
SWIRL_DIM = _hex("#926F34")
REC_FILL = _hex("#D2AC47")
REC_FILL_LIVE = _hex("#AE8625")
REC_DOT = _hex("#E23B3B")


def _font(px: int, fallback: bool = False):
    kind = "fallback" if fallback else "mahi"
    key = (max(10, px), kind)
    cached = _FONTS.get(key)
    if cached is not None:
        return cached
    path = FALLBACK_FONT if fallback else FONT
    if not Path(path).is_file():
        path = str(Path.home() / "Library" / "Fonts" / "Mahifont-Regular.ttf")
        if fallback or not Path(path).is_file():
            path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(path, key[0])
    _FONTS[key] = font
    return font


def _runs(text: str):
    if not text:
        return
    use_mahi = text[0] in _MAHI
    buf = text[0]
    for ch in text[1:]:
        ok = ch in _MAHI
        if ok == use_mahi:
            buf += ch
        else:
            yield buf, use_mahi
            buf, use_mahi = ch, ok
    yield buf, use_mahi


def _px(scale: float) -> int:
    return max(16, int(round(52 * scale)))


def _text_size(text: str, px: int, script: bool = False) -> tuple[int, int]:
    w, h = 0.0, 1
    for chunk, mahi in _runs(text):
        font = _font(px, fallback=not mahi)
        box = font.getbbox(chunk)
        w += font.getlength(chunk)
        h = max(h, box[3] - box[1])
    return max(1, int(w)), max(1, int(h))


def _label(img, text, xy, scale, color, thick=1, script=False, bold=False) -> None:
    _PENDING.append((text, int(xy[0]), int(xy[1]), _px(scale), color, bold))


def _flush(canvas: np.ndarray) -> np.ndarray:
    if not _PENDING:
        return canvas
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    shadow = (22, 12, 22)
    for text, x, y, px, color, bold in _PENDING:
        fill = (color[2], color[1], color[0])
        cx = x
        for chunk, mahi in _runs(text):
            font = _font(px, fallback=not mahi)
            draw.text((cx + 1, y + 1), chunk, font=font, fill=shadow, anchor="ls")
            draw.text((cx, y), chunk, font=font, fill=fill, anchor="ls")
            if bold:
                draw.text((cx + 1, y), chunk, font=font, fill=fill, anchor="ls")
            cx += font.getlength(chunk)
    _PENDING.clear()
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def compose(cam, gesture, slots: list[str], selected: int, editing: bool, valid: list[bool] | None = None, visible: list[int] | None = None, size: tuple[int, int] = (1280, 860), rec: bool = False):
    _PENDING.clear()
    W, H = max(int(size[0]), 400), max(int(size[1]), 300)
    s = min(W / 1280, H / 860)
    shown = visible or list(range(6))
    n = max(len(shown), 1)
    d = int(108 * s)
    gap = int(max(12, 22 * s))
    title_h = int(52 * s)
    hint_h = int(28 * s)
    pad = int(16 * s)
    cam_gap = int(52 * s)
    need = title_h + d + hint_h + pad + cam_gap
    if need > H * 0.42:
        shrink = (H * 0.42) / need
        d, title_h, hint_h, gap, pad, cam_gap = [int(v * shrink) for v in (d, title_h, hint_h, gap, pad, cam_gap)]
        s *= shrink
    max_row = W - int(48 * max(s, 0.5))
    if n * d + (n - 1) * gap > max_row:
        d = max(36, (max_row - (n - 1) * gap) // n)
    bottom = title_h + d + hint_h + pad + cam_gap
    gutter = min(int(max(168, 200 * s)), int(W * 0.24))
    top = int(max(10, 14 * s))
    right = int(max(12, 18 * s))
    max_w = max(2, W - gutter - right)
    max_h = max(2, H - top - bottom)

    ch0, cw0 = cam.shape[:2]
    scale = min(max_w / cw0, max_h / ch0)
    cw, ch = max(2, int(cw0 * scale)), max(2, int(ch0 * scale))
    cam = cv2.resize(cam, (cw, ch), interpolation=cv2.INTER_LINEAR)
    cx = gutter + (max_w - cw) // 2
    cy = top
    cam_bottom = cy + ch

    row_w = n * d + (n - 1) * gap
    g_step, g_extra, g_gap = int(30 * s), int(22 * s), int(16 * s)
    guide_h = int(32 * s) + n * g_step + 2 * g_gap + 6 * g_extra + int(48 * s)
    title_y = cam_bottom + cam_gap + int(10 * s)
    circ_y = cam_bottom + cam_gap + title_h
    blocked = [
        (int(max(8, 10 * s)), cy, gutter - int(6 * s), cy + guide_h),
        (cx - 8, cy - 8, cx + cw + 8, cy + ch + 8),
        (W // 2 - int(220 * s), title_y - int(22 * s), W // 2 + int(220 * s), title_y + int(10 * s)),
        ((W - row_w) // 2 - 14, circ_y - int(6 * s), (W + row_w) // 2 + 14, circ_y + d + int(8 * s)),
        (int(W * 0.16), H - hint_h - 8, int(W * 0.84), H),
    ]
    global _DECOR_KEY, _DECOR_IMG
    key = (W, H, n, gutter, cw, ch, cx, cy)
    if _DECOR_KEY != key or _DECOR_IMG is None:
        bg = np.full((H, W, 3), BG, np.uint8)
        _decor(bg, W, H, s, blocked)
        _DECOR_IMG, _DECOR_KEY = bg, key
    canvas = _DECOR_IMG.copy()

    _paste_round(canvas, cam, cx, cy, r=max(8, int(22 * s)), border=STROKE)
    _note(canvas, gesture, cx, cy, cw, ch, s)

    rec_rect = _guide(canvas, gesture.index, cy, s, H, gutter, hint_h, slots, shown, rec)
    rects, title = _circles(canvas, W, H, s, d, gap, title_h, slots, selected, editing, gesture.index, cam_bottom, valid or [True] * 6, shown, cam_gap)
    return _flush(canvas), rects, title, rec_rect


def _mix(stops: list[tuple[int, int, int]], t: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    n = len(stops) - 1
    x = t * n
    i = min(int(x), n - 1)
    f = x - i
    a, b = stops[i], stops[i + 1]
    return tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))


def _gold_ring(canvas, cx: int, cy: int, radius: int, thick: int, palette: list[tuple[int, int, int]]) -> None:
    steps = 80
    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        a0, a1 = t0 * math.tau, t1 * math.tau
        color = _mix(palette, t0)
        p0 = (int(cx + radius * math.cos(a0)), int(cy + radius * math.sin(a0)))
        p1 = (int(cx + radius * math.cos(a1)), int(cy + radius * math.sin(a1)))
        cv2.line(canvas, p0, p1, color, thick, cv2.LINE_AA)


def _star(canvas, cx, cy, r, color) -> None:
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.38
        pts.append((int(cx + rad * math.cos(ang)), int(cy + rad * math.sin(ang))))
    cv2.fillPoly(canvas, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def _swirl(canvas, cx, cy, r, color, thick=2, turns=2.15, flip=False) -> None:
    n = 90
    pts = []
    sign = -1 if flip else 1
    for i in range(n):
        t = i / (n - 1)
        ang = sign * t * turns * math.tau
        rad = r * (0.06 + 0.94 * t)
        pts.append((int(cx + rad * math.cos(ang)), int(cy + rad * math.sin(ang))))
    cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, thick, cv2.LINE_AA)


def _free(x, y, r, blocked) -> bool:
    for a, b, c, d in blocked:
        if x + r >= a and x - r <= c and y + r >= b and y - r <= d:
            return False
    return True


def _decor(canvas, W: int, H: int, s: float, blocked) -> None:
    swirls = (
        (0.05, 0.86, 34, SWIRL, True), (0.94, 0.84, 42, SWIRL_DIM, False),
        (0.96, 0.14, 34, SWIRL, False), (0.97, 0.40, 32, SWIRL, True),
        (0.95, 0.62, 28, SWIRL_DIM, False), (0.88, 0.28, 26, SWIRL, True),
        (0.91, 0.52, 24, SWIRL_DIM, False), (0.04, 0.08, 22, SWIRL_DIM, True),
    )
    stars = (
        (0.08, 0.78, 12, STAR), (0.03, 0.92, 9, STAR_DIM), (0.90, 0.80, 15, STAR),
        (0.96, 0.92, 10, STAR_DIM), (0.93, 0.06, 14, STAR), (0.97, 0.24, 9, STAR_DIM),
        (0.78, 0.035, 10, STAR), (0.96, 0.50, 13, STAR), (0.98, 0.70, 9, STAR_DIM),
        (0.86, 0.92, 9, STAR_DIM), (0.70, 0.03, 8, STAR), (0.84, 0.12, 11, STAR),
        (0.89, 0.36, 8, STAR_DIM), (0.99, 0.58, 10, STAR), (0.82, 0.88, 12, STAR),
        (0.74, 0.94, 7, STAR_DIM), (0.18, 0.04, 9, STAR), (0.02, 0.18, 8, STAR_DIM),
        (0.87, 0.68, 10, STAR), (0.93, 0.76, 7, STAR_DIM), (0.66, 0.06, 7, STAR_DIM),
        (0.02, 0.52, 8, STAR),
    )
    pad_star, pad_swirl = 22, 20
    for xf, yf, r, color, flip in swirls:
        x, y, rad = int(xf * W), int(yf * H), max(12, int(r * s))
        if _free(x, y, rad + pad_swirl, blocked):
            _swirl(canvas, x, y, rad, color, max(2, int(2.4 * s)), 2.25, flip)
    for xf, yf, r, color in stars:
        x, y, rad = int(xf * W), int(yf * H), max(6, int(r * s))
        if _free(x, y, rad + pad_star, blocked):
            _star(canvas, x, y, rad, color)
    rng = np.random.default_rng(11)
    for _ in range(420):
        x, y = int(rng.random() * W), int(rng.random() * H)
        if not _free(x, y, 8, blocked):
            continue
        color = STAR if rng.random() > 0.42 else STAR_DIM
        cv2.circle(canvas, (x, y), 1 if rng.random() > 0.28 else max(1, int(1.6 * s)), color, -1, cv2.LINE_AA)
    for _ in range(28):
        x, y = int(rng.random() * W), int(rng.random() * H)
        rad = max(3, int((3.5 + rng.random() * 2.5) * s))
        if _free(x, y, rad + 8, blocked):
            _star(canvas, x, y, rad, STAR if rng.random() > 0.35 else STAR_DIM)


def _chord_line(k: int, n: int, name: str) -> str:
    if n >= 6 and k == n - 1:
        left = "pinch"
    else:
        count = k + 1
        left = "1 finger" if count == 1 else f"{count} fingers"
    return f"{left} = {name}"


def _guide(canvas, playing: int, cam_y: int, s: float, H: int, gutter: int, hint_h: int, slots: list[str], shown: list[int], rec: bool) -> tuple[int, int, int, int]:
    r = max(8, int(22 * s))
    pad_x, pad_y = int(16 * s), int(16 * s)
    body, sub = 0.48 * s, 0.36 * s
    step, extra, gap = int(30 * s), int(22 * s), int(16 * s)
    rec_h = int(40 * s)
    lines = [_chord_line(k, len(shown), (slots[i].strip() or str(i + 1))) for k, i in enumerate(shown)]
    x1 = int(max(8, 10 * s))
    text_w = max(
        max(_text_size(line, _px(body))[0] for line in lines) if lines else 40,
        max(_text_size(line, _px(sub))[0] for line in HINTS + KEYS),
        _text_size("STOP", _px(0.5 * s))[0] + rec_h,
    )
    x2 = min(gutter - int(6 * s), x1 + text_w + 2 * pad_x)
    inner = max(40, x2 - x1 - 2 * pad_x)
    if text_w > inner:
        shrink = inner / text_w
        body, sub = body * shrink, sub * shrink
    n_chords = max(len(shown), 1)
    block = n_chords * step + gap + len(HINTS) * extra + gap + extra + gap + rec_h
    y1 = cam_y
    floor = H - hint_h - int(8 * s)
    y2 = y1 + pad_y + block + pad_y
    if y2 > floor:
        room = max(80, floor - y1 - 2 * pad_y - rec_h)
        step = max(13, (room - 2 * gap) // max(1, n_chords + len(HINTS) + 1))
        extra = step
        block = n_chords * step + gap + len(HINTS) * extra + gap + extra + gap + rec_h
        y2 = min(floor, y1 + pad_y + block + pad_y)
    _fill_round(canvas, x1, y1, x2, y2, FILL, r)
    _stroke_round(canvas, x1, y1, x2, y2, STROKE, r, 2)
    x = x1 + pad_x
    y = y1 + pad_y + _text_size(lines[0] if lines else "1", _px(body))[1]
    for k, (line, i) in enumerate(zip(lines, shown)):
        on = playing == i + 1
        _label(canvas, line, (x, y + k * step), body, PLAY if on else INK, thick=1 + int(on))
    ry = y + n_chords * step + gap
    for i, line in enumerate(HINTS):
        _label(canvas, line, (x, ry + i * extra), sub, HINT)
    ky = ry + len(HINTS) * extra + gap
    _label(canvas, KEYS[0], (x, ky), sub, HINT)
    bx1, by1 = x1 + pad_x, ky + extra + gap // 2
    bx2, by2 = x2 - pad_x, min(y2 - pad_y, by1 + rec_h)
    fill = REC_FILL_LIVE if rec else REC_FILL
    _fill_round(canvas, bx1, by1, bx2, by2, fill, r)
    _stroke_round(canvas, bx1, by1, bx2, by2, SWIRL_DIM, r, 2)
    label = "STOP" if rec else "REC"
    size = _text_size(label, _px(0.48 * s))
    mid = (by1 + by2) // 2 + int(5 * s)
    cv2.circle(canvas, (bx1 + int(18 * s), mid), max(5, int(7 * s)), REC_DOT, -1, cv2.LINE_AA)
    _label(canvas, label, (bx1 + int(32 * s), mid + size[1] // 3), 0.48 * s, BG, bold=True)
    return (bx1, by1, bx2, by2)


def _circles(canvas, W, H, s, d, gap, title_h, slots, selected, editing, playing, cam_bottom: int, valid: list[bool], shown: list[int], cam_gap: int = 0):
    title = "Enter chords below"
    ts = 0.82 * s
    title_size = _text_size(title, _px(ts), True)
    tx, ty = (W - title_size[0]) // 2, cam_bottom + cam_gap + int(10 * s)
    _label(canvas, title, (tx, ty), ts, INK, script=True, bold=True)
    title_rect = (tx, ty - title_size[1], tx + title_size[0], ty + 6)

    n = max(len(shown), 1)
    total = n * d + (n - 1) * gap
    x0 = (W - total) // 2
    mid_y = cam_bottom + cam_gap + title_h + d // 2
    if mid_y + d // 2 > H - int(28 * s):
        mid_y = H - int(28 * s) - d // 2
    blink = editing and int(time.time() * 2) % 2 == 0
    rects = [(0, 0, 0, 0)] * 6
    for k, i in enumerate(shown):
        cx = x0 + k * (d + gap) + d // 2
        x1, y1, x2, y2 = cx - d // 2, mid_y - d // 2, cx + d // 2, mid_y + d // 2
        rects[i] = (x1, y1, x2, y2)
        live, focus = playing == i + 1, selected == i
        palette = GOLD_PLAY if live else GOLD_IDLE
        thick = 4 + int(live) if live or focus else 3
        cv2.circle(canvas, (cx, mid_y), d // 2, FILL, -1, cv2.LINE_AA)
        _gold_ring(canvas, cx, mid_y, d // 2, thick, palette)
        text = slots[i]
        if focus and blink:
            text += "|"
        if slots[i].strip():
            _label(canvas, str(i + 1), (cx - int(6 * s), y1 + int(28 * s)), 0.4 * s, MUTED)
            _fit(canvas, text, (x1 + 8, y1, x2 - 8, y2), INK if valid[i] else (90, 90, 210), 0.62 * s, bold=True)
        else:
            label = text if text else str(i + 1)
            size = _text_size(label, _px(0.9 * s))
            _label(
                canvas, label, (cx - size[0] // 2, mid_y + size[1] // 4), 0.9 * s,
                STAR if focus else MUTED, bold=bool(text),
            )
    hint = "space / right arrow = next   Enter hides empty"
    hint_size = _text_size(hint, _px(0.4 * s))
    _label(canvas, hint, ((W - hint_size[0]) // 2, H - int(16 * s) - 2), 0.4 * s, HINT)
    return rects, title_rect


def _fit(canvas, text: str, box, color, scale: float, bold: bool = False) -> None:
    x1, y1, x2, y2 = box
    shown, use = text, scale
    for try_s in (scale, scale * 0.87, scale * 0.76, scale * 0.65):
        size = _text_size(shown, _px(try_s))
        if size[0] <= (x2 - x1) - 6:
            use = try_s
            break
    size = _text_size(shown, _px(use))
    while shown and size[0] > (x2 - x1) - 6:
        shown = shown[:-1]
        size = _text_size(shown, _px(use))
    _label(canvas, shown, (x1 + (x2 - x1 - size[0]) // 2, (y1 + y2) // 2 + size[1] // 4), use, color, bold=bold)


def _note(canvas, gesture, x, y, w, h, s: float) -> None:
    label = gesture.note if gesture.playing else "waiting for your hands..."
    scale = (1.15 if gesture.playing else 0.5) * s
    size = _text_size(label, _px(scale))
    _label(canvas, label, (x + (w - size[0]) // 2, y + h - int(22 * s)), scale, INK, thick=2 if gesture.playing else 1)


def _fill_round(img, x1, y1, x2, y2, color, r) -> None:
    r = int(max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _stroke_round(img, x1, y1, x2, y2, color, r, t) -> None:
    r = int(max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, cv2.LINE_AA)


def _paste_round(dst, src, x, y, r, border) -> None:
    h, w = src.shape[:2]
    x = max(0, min(x, dst.shape[1] - 2))
    y = max(0, min(y, dst.shape[0] - 2))
    h = min(h, dst.shape[0] - y)
    w = min(w, dst.shape[1] - x)
    if h < 2 or w < 2:
        return
    src = src[:h, :w]
    mask = np.zeros((h, w), np.uint8)
    _fill_round(mask, 0, 0, w, h, 255, r)
    roi = dst[y : y + h, x : x + w]
    roi[:] = np.where(mask[:, :, None] > 0, src, roi)
    _stroke_round(dst, x, y, x + w, y + h, border, r, 2)
