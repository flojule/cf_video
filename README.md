# cf_video

Tracks 3 CrazyFlie drones (cf1, cf2, cf3) and a hanging payload in a video.
Uses foreground blob detection with Hungarian assignment to maintain identities across frames.

[debug video](https://github.com/user-attachments/assets/e67bd61e-d37c-42d2-9480-7630b7cadb73)

[persistent trails](https://github.com/user-attachments/assets/3c89aa1e-8dce-472c-b3af-a6215bce1133)

[transient trails](https://github.com/user-attachments/assets/e7789a11-9e12-441e-bdb5-66d82cba9e9f)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**1. Select ROIs and tracking window**

```bash
python src/setup.py
```

Interactively pick bounding boxes for each agent and mark the first/last frame of interest.
Saves `src/rois.json`.

**2. Run tracking**

```bash
python src/annotate.py
```

Reads `input/crazyflo.mp4`, writes outputs to `output/`.

## Outputs

| File | Description |
|---|---|
| `crazyflo_path.mp4` | Tracking overlay (bbox + trail per agent) |
| `crazyflo_path_debug.mp4` | Debug view with blob detections and state info |
| `crazyflo_path_persistent.mp4` | Clean video with trails that accumulate over time |
| `crazyflo_path_transient.mp4` | Same trails but fading — window sized to less than one oscillation |
| `track_and_annotate.log` | Run log |

## Configuration

Key constants at the top of `src/annotate.py`:

| Constant | Default | Description |
|---|---|---|
| `VIDEO_IN` | `input/crazyflo.mp4` | Input video |
| `VIDEO_OUT` | `output/crazyflo_path.mp4` | Primary output path (other outputs derive from this stem) |
| `trail_start_sec` | `2` | Seconds from start before trails appear |
| `TRAIL_COLOR` | per-agent | BGR colors: cf1=red, cf2=green, cf3=blue, payload=dark grey |
| `MAX_ASSIGN_DIST` | `260` | Max px distance for Hungarian assignment |
| `AGENT_Y_CLAMP` | `{"payload": 120}` | Max vertical deviation allowed for payload in assignment |
| `SKIP_FFMPEG_DEFAULT` | `True` | Set to `False` to auto-convert outputs to webm via ffmpeg |
