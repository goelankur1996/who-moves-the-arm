"""Ghost overlay: the live scene camera blended with a layout's reference image, for placing pieces.

    python scripts/ghost.py --set eval              # E1-01 ... E3-10
    python scripts/ghost.py --set demo --start D-21

Keys:  n or space  next layout      p  previous layout
       a  cycle the blend (50/50, live only, reference only)      q or Esc  quit

Put each piece on its outline (the blue line), turned the same way; each container's rim on its circle.
"""
import argparse
import json

import cv2
import numpy as np

LAYOUTS = "layouts/layouts.json"
WIN = "ghost"
# Outer top rims of the containers on their spots: centre u, v and half-widths across and down,
# pixels, read off calib/containers_now.jpg (26 Nov).
RIMS = {"pink_tumbler": (255.6, 313.0, 26.4, 24.8), "mug": (521.0, 250.5, 22.3, 19.8)}


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(10):
        ok, f = cap.read()
    if not ok or f.shape[:2] != (480, 640):
        raise SystemExit(f"camera {index}: no 640x480 frames (got {None if not ok else f.shape})")
    return cap


def outlines(ref, base):
    """Contours of the drawn pieces: where the reference differs from the empty base frame."""
    d = cv2.absdiff(ref, base).max(axis=2)
    m = (d > 40).astype(np.uint8) * 255
    m[:28] = 0                                                         # the title band
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drops the thin labels
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cs if cv2.contourArea(c) > 40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["eval", "demo"], required=True)
    ap.add_argument("--start", help="layout id to start at, e.g. E2-01 or D-21")
    ap.add_argument("--camera", type=int, default=0, help="scene camera index (calibration v2 used 0)")
    args = ap.parse_args()

    data = json.load(open(LAYOUTS))
    layouts = [L for L in data["layouts"] if L["set"] == args.set]
    base = cv2.imread(data["meta"]["base_image"])
    i = next((k for k, L in enumerate(layouts) if L["layout_id"] == args.start), 0)
    cap = open_camera(args.camera)
    modes = [0.5, 0.0, 1.0]  # weight of the reference image
    mode, cached = 0, None

    while True:
        L = layouts[i]
        if cached is None or cached[0] != i:
            ref = cv2.imread(L["image"])
            cached = (i, ref, outlines(ref, base))
            print(f"{L['layout_id']} ({i + 1}/{len(layouts)}): {L['instruction']}")
        _, ref, cs = cached
        ok, live = cap.read()
        if not ok:
            continue
        w = modes[mode]
        view = cv2.addWeighted(ref, w, live, 1 - w, 0)
        cv2.drawContours(view, cs, -1, (255, 90, 0), 1, cv2.LINE_AA)
        for u, v, a, b in RIMS.values():
            cv2.ellipse(view, (round(u), round(v)), (round(a), round(b)), 0, 0, 360, (255, 90, 0), 1, cv2.LINE_AA)
        cv2.rectangle(view, (0, 0), (640, 26), (255, 255, 255), -1)
        cv2.putText(view, f"{L['layout_id']} {i + 1}/{len(layouts)}  {L['instruction']}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(WIN, cv2.resize(view, (1280, 960)))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord(" ")):
            i = min(i + 1, len(layouts) - 1)
        elif key == ord("p"):
            i = max(i - 1, 0)
        elif key == ord("a"):
            mode = (mode + 1) % len(modes)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
