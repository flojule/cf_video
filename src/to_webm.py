"""to_webm.py — convert all MP4 files in output/ to WebM (VP9) in output/webm/.

Usage:
    python src/to_webm.py              # scans output/
    python src/to_webm.py path/to/dir  # scans a custom directory
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

INPUT_DIR  = Path("output")
OUTPUT_DIR = INPUT_DIR / "webm"
BITRATE    = "2M"


def convert(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-c:v", "libvpx-vp9", "-b:v", BITRATE, str(dst)],
        capture_output=True, timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src.name}: {r.stderr[:300].decode(errors='replace')}")


def main(scan_dir: Path = INPUT_DIR) -> None:
    mp4s = sorted(scan_dir.glob("*.mp4"))
    if not mp4s:
        print(f"No MP4 files found in {scan_dir}")
        return

    out_dir = scan_dir / "webm"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in mp4s:
        dst = out_dir / src.with_suffix(".webm").name
        print(f"  {src.name} → webm/{dst.name} …", end=" ", flush=True)
        convert(src, dst)
        print("done")

    print(f"\nConverted {len(mp4s)} file(s) → {out_dir}")


if __name__ == "__main__":
    scan = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_DIR
    main(scan)
