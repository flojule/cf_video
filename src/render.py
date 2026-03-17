"""render.py — post-process trail videos from tracking data exported by track.py.

Usage:
    python src/render.py                              # uses default data file
    python src/render.py output/crazyflo_path_tracking.json

Outputs (MP4, next to the JSON file):
    <stem>_persistent.mp4
    <stem>_transient.mp4

Any trail property can be overridden via the RENDER_* env vars or by editing
the OVERRIDES dict at the top of this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

# ── Optional per-run overrides (None = use value from JSON) ────────────────────
OVERRIDES: dict = {
    # "trail_thickness": 3,
    # "alpha": 0.6,
    # "trail_window": 200,
    # "smooth_trails": True,
    # "trail_color": {"cf1": [0, 0, 255], "cf2": [0, 255, 0], "cf3": [255, 0, 0], "payload": [50, 50, 50]},
}

DEFAULT_DATA_FILE = "output/crazyflo_path_tracking.json"


# ── Rendering helpers (mirrors annotate.py) ────────────────────────────────────

def smooth_pts(pts: list) -> list:
    n = len(pts)
    if n < 5:
        return list(pts)
    win = min(15, n if n % 2 == 1 else n - 1)
    xs = savgol_filter([p[0] for p in pts], win, 3)
    ys = savgol_filter([p[1] for p in pts], win, 3)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def blend_trail(frame: np.ndarray, canvas: np.ndarray, alpha: float) -> np.ndarray:
    mask = canvas.any(axis=2)
    out  = frame.copy()
    out[mask] = cv2.addWeighted(frame, 1 - alpha, canvas, alpha, 0)[mask]
    return out



def render(data_file: str = DEFAULT_DATA_FILE) -> None:
    with open(data_file) as f:
        d = json.load(f)

    # Apply overrides
    for k, v in OVERRIDES.items():
        if v is not None:
            d[k] = v

    video_in        = d["video_in"]
    fps             = float(d["fps"])
    total           = int(d["total_frames"])
    W               = int(d["width"])
    H               = int(d["height"])
    trail_start_sec = float(d["trail_start_sec"])
    trail_end_sec   = float(d["trail_end_sec"])
    trail_color     = {k: tuple(v) for k, v in d["trail_color"].items()}
    thickness       = int(d["trail_thickness"])
    alpha           = float(d["alpha"])
    trail_window    = int(d["trail_window"])
    smooth          = bool(d["smooth_trails"])

    # Trails: {agent: [(frame_idx, (x, y)), ...]} sorted by frame
    trails: dict[str, list[tuple[int, tuple[int, int]]]] = {
        name: sorted((int(fi), tuple(pt)) for fi, pt in pts.items())
        for name, pts in d["trails"].items()
    }

    trail_start_frame = int(trail_start_sec * fps)
    trail_end_frame   = total - int(trail_end_sec * fps)

    # Dynamic transient window from payload speed
    payload_seq = trails.get("payload", [])
    if len(payload_seq) >= 10:
        steps = [
            ((payload_seq[k+1][1][0] - payload_seq[k][1][0])**2
             + (payload_seq[k+1][1][1] - payload_seq[k][1][1])**2)**0.5
            for k in range(len(payload_seq) - 1)
        ]
        valid = [s for s in steps if s > 0.1]
        if valid:
            trail_window = max(20, int(W / float(np.median(valid)) * 0.45))

    stem   = Path(data_file).stem.removesuffix("_tracking")
    parent = Path(data_file).parent
    out_p  = parent / (stem + "_persistent.mp4")
    out_t  = parent / (stem + "_transient.mp4")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    head   = {name: 0 for name in trails}

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        sys.exit(f"Cannot open {video_in}")

    wr_p = cv2.VideoWriter(str(out_p), fourcc, fps, (W, H))
    wr_t = cv2.VideoWriter(str(out_t), fourcc, fps, (W, H))

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        show_trail  = trail_start_frame <= fi <= trail_end_frame
        drain_trail = (not show_trail) and (fi <= trail_end_frame + trail_window)

        if not show_trail and not drain_trail:
            wr_p.write(frame)
            wr_t.write(frame)
            continue

        # Advance head pointers only while actively tracking
        if show_trail:
            for name, seq in trails.items():
                while head[name] < len(seq) and seq[head[name]][0] <= fi:
                    head[name] += 1

        # ── Persistent ────────────────────────────────────────────────────────
        if show_trail:
            cp = np.zeros((H, W, 3), dtype=np.uint8)
            for name, seq in trails.items():
                pts  = [pt for f, pt in seq[:head[name]] if f >= trail_start_frame]
                draw = smooth_pts(pts) if (smooth and len(pts) >= 2) else pts
                if len(draw) >= 2:
                    cv2.polylines(cp, [np.array(draw, dtype=np.int32)],
                                  False, trail_color[name], thickness)
            wr_p.write(blend_trail(frame, cp, alpha))
        else:
            wr_p.write(frame)

        # ── Transient (shooting star: thick bright head → thin faded tail) ───
        # During drain_trail, heads are frozen so the window scrolls the tail
        # away at the normal rate — no sudden disappearance.
        ct = np.zeros((H, W, 3), dtype=np.uint8)
        for name, seq in trails.items():
            window = [(f, pt) for f, pt in seq[:head[name]]
                      if fi - trail_window <= f]
            if len(window) < 2:
                continue
            draw = smooth_pts([pt for _, pt in window]) if smooth else [pt for _, pt in window]
            for k in range(len(draw) - 1):
                age = fi - window[k + 1][0]
                w   = max(0.0, 1.0 - age / trail_window)
                col = tuple(int(c * w) for c in trail_color[name])
                lw  = max(1, round(thickness * w ** 0.5))
                cv2.line(ct, draw[k], draw[k + 1], col, lw)
        wr_t.write(blend_trail(frame, ct, alpha))

        if fi % 30 == 0:
            sys.stdout.write(f"\r  {100*fi/total:.0f}%")
            sys.stdout.flush()

    sys.stdout.write("\r  100%\n")
    sys.stdout.flush()
    cap.release()
    wr_p.release()
    wr_t.release()

    print(f"Done.\n  {out_p}\n  {out_t}")


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_FILE)
