#!/usr/bin/env python3
# setup.py (renamed from select_rois.py)
"""Select start/end frames and ROIs for tracking (interactive).

Produces `rois.json` with structure:
{
  "start_frame": <int>,
  "end_frame": <int>,
  "rois": { "agent": [x,y,w,h], ... }
}
"""
import json
import cv2

VIDEO = "input/crazyflo.mp4"         # ← your input
OUTPUT = "src/rois.json"

AGENTS = ["cf1", "cf2", "cf3", "payload"]
COLORS = {
    "cf1":     (255, 255,   0),
    "cf2":     (255,   0, 255),
    "cf3":     (  0, 255, 255),
    "payload": (255, 255, 255),
}


def main():

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    print("Use LEFT/RIGHT arrows to move 10 frames, a/d to move 1 frame, SPACE/ENTER to confirm.")

    # --- Select start frame ---
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
        display = frame.copy()
        cv2.putText(display, f"Frame {frame_idx+1}/{total} (START)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Select starting frame", display)
        key = cv2.waitKey(0) & 0xFF
        if key in [32, 13]:  # SPACE or ENTER
            break
        elif key == 81:  # LEFT arrow
            frame_idx = max(0, frame_idx - 10)
        elif key == 83:  # RIGHT arrow
            frame_idx = min(total-1, frame_idx + 10)
        elif key == ord('a'):
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('d'):
            frame_idx = min(total-1, frame_idx + 1)
        elif key == 27:  # ESC
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
    start_frame = frame_idx
    cv2.destroyAllWindows()

    # --- Select end frame ---
    frame_idx = start_frame
    print("Now select END frame (should be >= start frame). Same controls.")
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            frame_idx = start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
        display = frame.copy()
        cv2.putText(display, f"Frame {frame_idx+1}/{total} (END)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)
        cv2.imshow("Select end frame", display)
        key = cv2.waitKey(0) & 0xFF
        if key in [32, 13]:  # SPACE or ENTER
            if frame_idx < start_frame:
                print("End frame must be >= start frame!")
                continue
            break
        elif key == 81:  # LEFT arrow
            frame_idx = max(start_frame, frame_idx - 10)
        elif key == 83:  # RIGHT arrow
            frame_idx = min(total-1, frame_idx + 10)
        elif key == ord('a'):
            frame_idx = max(start_frame, frame_idx - 1)
        elif key == ord('d'):
            frame_idx = min(total-1, frame_idx + 1)
        elif key == 27:  # ESC
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
    end_frame = frame_idx
    cv2.destroyAllWindows()

    # Use the selected start frame for ROI selection
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read start frame {start_frame}")
        cap.release()
        return

    # Use the selected frame for ROI selection
    rois = {}
    for agent in AGENTS:
        color = COLORS[agent]
        display = frame.copy()
        cv2.putText(
            display,
            f"Draw box around: {agent}  (SPACE/ENTER to confirm, C to cancel)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
        roi = cv2.selectROI(f"Select {agent}", display, fromCenter=False)
        cv2.destroyAllWindows()
        if roi == (0, 0, 0, 0):
            print(f"Skipping {agent}")
            continue
        rois[agent] = list(roi)   # (x, y, w, h)
        print(f"{agent}: {roi}")


    # Save ROIs and frame window
    out_data = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "rois": rois
    }
    with open(OUTPUT, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
