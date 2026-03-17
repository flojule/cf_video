# Comet

Motion-trail visualisation tool for tracking multiple objects in video — including well-defined objects (e.g. drones, vehicles, animals) and harder-to-see ones (e.g. a small payload). Uses background subtraction + Hungarian assignment to maintain object identities across frames, then renders clean trail videos.

[debug video](https://github.com/user-attachments/assets/a0e2a55d-9e8b-4190-9f15-e77fb74a519f)

[transient trails](https://github.com/user-attachments/assets/e33fe741-0fd7-465e-98d2-c9d5fde95972)

[persistent trails](https://github.com/user-attachments/assets/be91b6bf-33b3-4584-b761-8da9fd4e519d)

## Workflow

```
input video
    │
    ▼
pick.py          ← interactively select objects and frame range
    │ rois.json
    ▼
track.py         ← background subtraction + Hungarian tracking
    │ *_tracking.json + *_debug.mp4
    ▼
render.py        ← render persistent and transient trail videos
    │ *_persistent.mp4 + *_transient.mp4
    ▼
to_webm.py       ← (optional) batch-convert output/*.mp4 → output/webm/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**1. Pick objects and frame range** — `src/pick.py`

```bash
python src/pick.py
```

Opens an interactive window. Use arrow keys / `a` / `d` to scrub the video, then press Space/Enter to confirm each step:

1. **Start frame** — first frame of interest (objects should be in their starting positions)
2. **End frame** — last frame of interest
3. **Bounding boxes** — draw a box around each object to track (one at a time)

Objects are defined in the `AGENTS` list at the top of `pick.py`. Each agent can represent any moving object — label them to suit your footage. The tool handles two distinct detection modes under the hood:

- **High-contrast objects** (well-lit, distinct from background) — tracked with a standard foreground mask
- **Low-contrast objects** (small, dim, or occluded) — additionally detected with a more sensitive mask restricted to the object's expected spatial band, so they don't get lost when the standard threshold misses them

Saves `src/rois.json`.

**2. Track** — `src/track.py`

```bash
python src/track.py
```

Reads `input/` video + `src/rois.json`. Builds a background model from pre-motion frames, then tracks all objects frame-by-frame. Gap periods (when an object is temporarily undetected) are filled using a bidirectional corridor search.

Writes to `output/`:
- `*_debug.mp4` — annotated video showing bounding boxes, labels, and trails
- `*_tracking.json` — full per-frame coordinate log for all objects

**3. Render trails** — `src/render.py`

```bash
python src/render.py
```

Reads the tracking JSON and produces two clean videos:

| Output | Description |
|---|---|
| `*_persistent.mp4` | Trails accumulate from start to end |
| `*_transient.mp4` | Shooting-star style: thick at the current position, tapering and fading over ~¼ orbit |

Trail appearance (color, thickness, alpha, window length) can be overridden via the `OVERRIDES` dict at the top of the file without re-running tracking.

**4. Convert to WebM** — `src/to_webm.py` *(optional)*

```bash
python src/to_webm.py
```

Batch-converts all `output/*.mp4` → `output/webm/*.webm` using VP9. Useful for web embedding.

## Configuration

Key constants in `src/track.py`:

| Constant | Default | Description |
|---|---|---|
| `VIDEO_IN` | `input/crazyflo.mp4` | Input video path |
| `VIDEO_OUT` | `output/crazyflo_path.mp4` | Output path stem |
| `AGENTS` | `["cf1","cf2","cf3","payload"]` | Object names (edit in `pick.py` too) |
| `TRAIL_COLOR` | per-agent BGR | Colors for each object |
| `MAX_ASSIGN_DIST` | `260` | Max px distance for Hungarian assignment |
| `AGENT_Y_CLAMP` | `{"payload": 120}` | Hard vertical gate for low-contrast objects |
| `CLAMP_WHEN_LOST` | `{"payload"}` | Agents that fall back to nearest blob when unmatched |
