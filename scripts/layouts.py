"""Generate the evaluation and demonstration layouts and a reference image for each.

    python scripts/layouts.py              # writes layouts/layouts.json and layouts/img/*.jpg
    python scripts/layouts.py --preview    # also writes layouts/allowed_area.jpg

Evaluation: 30 layouts (10 E1, 10 E2, 10 E3), seed 20261123 (PREREG).
Demonstrations: 80 layouts with training objects only, seed 20261125.
Seed 20261124 is reserved for the week 3 layouts.

Rules (PREREG deviation 8):
- The containers stay on their marks in every layout.
- Piece centres lie inside the area enclosed by the 9 points of calibration v2,
  at least EDGE_MARGIN_MM inside its edge, clear of the containers and of the arm at rest,
  and at least MIN_SEP_MM apart.
"""
import argparse
import json
import os

import cv2
import numpy as np

CALIB = "calib/touch_calib_v2.json"
BASE = "calib/layout_base_v2.jpg"  # scene frame at 640x480, arm at rest, containers on their marks, no piece
OUT = "layouts"
EVAL_SEED, DEMO_SEED = 20261123, 20261125

TRAIN = ["orange_cube", "blue_b", "green_d"]
HELD_OUT = ["red_c", "blue_star", "teal_triangle"]
CONTAINERS = ["pink_tumbler", "mug"]
ARMS = ["A", "B", "C", "C-ACT"]

NAMES = {
    "orange_cube": "the orange cube", "blue_b": "the blue B", "green_d": "the green D",
    "red_c": "the red C", "blue_star": "the blue star", "teal_triangle": "the teal triangle",
    "pink_tumbler": "the pink tumbler", "mug": "the mug",
}
# (shape, size in mm, BGR colour) used only to draw the reference images
SHAPES = {
    "orange_cube": ("square", 25, (0, 140, 255)),
    "blue_b": ("B", 30, (235, 200, 120)),
    "green_d": ("D", 30, (60, 225, 160)),
    "red_c": ("C", 30, (90, 100, 245)),
    "blue_star": ("star", 35, (200, 110, 30)),
    "teal_triangle": ("triangle", 30, (160, 160, 30)),
}

# Container marks: pixel centres in the scene camera, measured on the base frame (26 Nov).
# Tumbler: centre of the rim. Mug: centre of the orange base.
CONTAINER_PX = {"pink_tumbler": (254, 313), "mug": (501, 269)}
CONTAINER_CLEAR_MM = 80  # piece centre to container centre
EDGE_MARGIN_MM = 10      # piece centre inside the calibrated area's edge
MIN_SEP_MM = 70          # piece centre to piece centre (the open fingertips are 86 mm apart)
ARM_CLEAR_PX = 25        # clearance around anything that is not bare table in the base frame


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


class Geometry:
    def __init__(self, calib_path, base_path):
        c = json.load(open(calib_path))
        self.H = np.array(c["H_image_to_table_mm"])
        self.Hinv = np.linalg.inv(self.H)
        img = np.array(c["image_points"], float)
        self.hull_px = cv2.convexHull(img.astype(np.float32)).reshape(-1, 2)
        self.hull_mm = np.array([self.to_mm(p) for p in self.hull_px])
        self.base = cv2.imread(base_path)
        assert self.base is not None, base_path
        self.blocked = self._blocked_mask(self.base)
        self.cont_mm = {k: self.to_mm(v) for k, v in CONTAINER_PX.items()}

    def to_mm(self, p):
        v = self.H @ np.array([p[0], p[1], 1.0])
        return v[:2] / v[2]

    def to_px(self, q):
        v = self.Hinv @ np.array([q[0], q[1], 1.0])
        return v[:2] / v[2]

    @staticmethod
    def _blocked_mask(base):
        """Pixels that are not bare table (arm, cables, containers), grown by ARM_CLEAR_PX."""
        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 2] < 150) | (hsv[..., 1] > 70)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop specks
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ARM_CLEAR_PX + 1,) * 2)
        return cv2.dilate(m, k) > 0

    def _inside(self, q):
        """Inside the calibrated area by at least EDGE_MARGIN_MM (hull is convex)."""
        n = len(self.hull_mm)
        area = 0.5 * sum(cross(self.hull_mm[i], self.hull_mm[(i + 1) % n]) for i in range(n))
        s = np.sign(area)
        for i in range(n):
            a, b = self.hull_mm[i], self.hull_mm[(i + 1) % n]
            e = b - a
            if s * cross(e, q - a) / np.linalg.norm(e) < EDGE_MARGIN_MM:
                return False
        return True

    def allowed(self, q, taken=()):
        if not self._inside(q):
            return False
        if any(np.linalg.norm(q - c) < CONTAINER_CLEAR_MM for c in self.cont_mm.values()):
            return False
        if any(np.linalg.norm(q - t) < MIN_SEP_MM for t in taken):
            return False
        u, v = np.round(self.to_px(q)).astype(int)
        h, w = self.blocked.shape
        return 0 <= u < w and 0 <= v < h and not self.blocked[v, u]

    def sample(self, rng, taken):
        lo, hi = self.hull_mm.min(0), self.hull_mm.max(0)
        for _ in range(100000):
            q = rng.uniform(lo, hi)
            if self.allowed(q, taken):
                return q
        raise RuntimeError("no free spot; loosen the clearances")


def cycle(rng, items, n):
    """n items drawn from shuffled copies of `items`, so each appears as evenly as possible."""
    out = []
    while len(out) < n:
        out += [items[i] for i in rng.permutation(len(items))]
    return out[:n]


def pairs_balanced(rng, objects, n):
    """n (object, container) pairs: every pair at least once when n allows, and the two
    containers used equally often (to within one)."""
    pairs = [(o, c) for o in objects for c in CONTAINERS]
    while True:
        out = cycle(rng, pairs, n)
        k = sum(c == CONTAINERS[0] for _, c in out)
        if abs(2 * k - n) <= n % 2:
            return out


def make_layout(geo, rng, layout_id, set_name, condition, target, container, distractors, index):
    pieces, taken = [], []
    for obj, role in [(target, "target")] + [(d, "distractor") for d in distractors]:
        q = geo.sample(rng, taken)
        taken.append(q)
        u, v = geo.to_px(q)
        pieces.append({"id": obj, "role": role,
                       "table_mm": [round(float(q[0]), 1), round(float(q[1]), 1)],
                       "pixel": [round(float(u), 1), round(float(v), 1)],
                       "yaw_deg": round(float(rng.uniform(0, 360)), 1)})
    out = {"layout_id": layout_id, "set": set_name, "condition": condition,
           "instruction": f"put {NAMES[target]} in {NAMES[container]}",
           "target": target, "container": container, "pieces": pieces,
           "image": f"{OUT}/img/{layout_id}.jpg"}
    if set_name == "eval":
        out["arm_order"] = ARMS[index % 4:] + ARMS[:index % 4]
    return out


def eval_layouts(geo):
    rng = np.random.default_rng(EVAL_SEED)
    layouts = []
    for i, (o, c) in enumerate(pairs_balanced(rng, TRAIN, 10)):
        layouts.append(make_layout(geo, rng, f"E1-{i + 1:02d}", "eval", "E1", o, c, [], len(layouts)))
    for i, (o, c) in enumerate(pairs_balanced(rng, TRAIN, 10)):
        others = [t for t in TRAIN if t != o]
        layouts.append(make_layout(geo, rng, f"E2-{i + 1:02d}", "eval", "E2", o, c, others, len(layouts)))
    distractors = cycle(rng, TRAIN, 10)
    for i, (o, c) in enumerate(pairs_balanced(rng, HELD_OUT, 10)):
        layouts.append(make_layout(geo, rng, f"E3-{i + 1:02d}", "eval", "E3", o, c, [distractors[i]], len(layouts)))
    return layouts


def demo_layouts(geo):
    """80 demonstrations, training objects only: 40 alone, 20 with one distractor, 20 with two."""
    rng = np.random.default_rng(DEMO_SEED)
    n_dis = [0] * 40 + [1] * 20 + [2] * 20
    combos = pairs_balanced(rng, TRAIN, 40) + pairs_balanced(rng, TRAIN, 20) + pairs_balanced(rng, TRAIN, 20)
    order = rng.permutation(80)  # mix the three kinds through the recording sessions
    layouts, seen = [], {}
    for i, j in enumerate(order):
        o, c = combos[j]
        others = [t for t in TRAIN if t != o]
        if n_dis[j] == 1:  # alternate the single distractor per target, so each piece is used evenly
            seen[o] = seen.get(o, int(rng.integers(2))) + 1
            dis = [others[seen[o] % 2]]
        else:
            dis = list(rng.permutation(others))[:n_dis[j]]
        cond = ["D0", "D1", "D2"][n_dis[j]]
        layouts.append(make_layout(geo, rng, f"D-{i + 1:02d}", "demo", cond, o, c, dis, i))
    return layouts


# ---------- reference images ----------

def piece_patch(obj, ppm=6):
    """Draw a piece on a square patch, ppm pixels per mm, centred. Returns (mask, size_mm)."""
    shape, size, _ = SHAPES[obj]
    S = int(2 * size * ppm)
    m = np.zeros((S, S), np.uint8)
    c, r = S / 2, size * ppm / 2
    if shape == "square":
        cv2.rectangle(m, (int(c - r), int(c - r)), (int(c + r), int(c + r)), 255, -1)
    elif shape == "triangle":
        pts = [(c + r * np.cos(a), c + r * np.sin(a)) for a in np.radians([-90, 30, 150])]
        cv2.fillPoly(m, [np.int32(pts)], 255)
    elif shape == "star":
        pts = [(c + (r if k % 2 == 0 else r * 0.45) * np.cos(a), c + (r if k % 2 == 0 else r * 0.45) * np.sin(a))
               for k, a in enumerate(np.radians(np.arange(-90, 270, 36)))]
        cv2.fillPoly(m, [np.int32(pts)], 255)
    else:  # a letter
        scale, thick = 1.0, 1
        (tw, th), _ = cv2.getTextSize(shape, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        scale = 2 * r / th
        thick = max(2, int(scale * 4))
        (tw, th), _ = cv2.getTextSize(shape, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        cv2.putText(m, shape, (int(c - tw / 2), int(c + th / 2)), cv2.FONT_HERSHEY_DUPLEX, scale, 255, thick)
    return m, 2 * size


def draw_piece(img, geo, piece, ppm=6):
    mask, size_mm = piece_patch(piece["id"], ppm)
    S = mask.shape[0]
    yaw = np.radians(piece["yaw_deg"])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    x0, y0 = piece["table_mm"]
    # patch pixel -> table mm: centred, ppm px per mm, rotated by yaw
    A = np.eye(3)
    A[:2, :2] = R / ppm
    A[:2, 2] = [x0, y0] - R @ np.array([S / 2, S / 2]) / ppm
    M = geo.Hinv @ A
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_LINEAR) > 127
    colour = SHAPES[piece["id"]][2]
    img[warped] = colour
    edge = cv2.morphologyEx(warped.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    img[edge] = (40, 40, 40)
    u, v = np.int32(piece["pixel"])
    label = piece["id"].replace("_", " ") + (" *" if piece["role"] == "target" else "")
    cv2.putText(img, label, (u - 30, v + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)


def render(geo, layout):
    img = geo.base.copy()
    for k, (u, v) in CONTAINER_PX.items():
        cv2.drawMarker(img, (u, v), (160, 80, 0), cv2.MARKER_CROSS, 14, 1)
    for p in layout["pieces"]:
        draw_piece(img, geo, p)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(img, f"{layout['layout_id']}  {layout['instruction']}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def preview(geo):
    img = geo.base.copy()
    ok = np.zeros(geo.blocked.shape, bool)
    for v in range(0, ok.shape[0], 2):
        for u in range(0, ok.shape[1], 2):
            if geo.allowed(geo.to_mm((u, v))):
                ok[v:v + 2, u:u + 2] = True
    img[ok] = (0.6 * img[ok] + 0.4 * np.array([255, 160, 60])).astype(np.uint8)
    cv2.polylines(img, [np.int32(geo.hull_px)], True, (120, 60, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="also write layouts/allowed_area.jpg")
    args = ap.parse_args()
    geo = Geometry(CALIB, BASE)
    layouts = eval_layouts(geo) + demo_layouts(geo)
    os.makedirs(f"{OUT}/img", exist_ok=True)
    for L in layouts:
        cv2.imwrite(L["image"], render(geo, L), [cv2.IMWRITE_JPEG_QUALITY, 90])
    meta = {"calib": CALIB, "base_image": BASE, "eval_seed": EVAL_SEED, "demo_seed": DEMO_SEED,
            "week3_seed_reserved": 20261124, "container_px": CONTAINER_PX,
            "container_table_mm": {k: [round(float(x), 1) for x in v] for k, v in geo.cont_mm.items()},
            "clearances": {"container_mm": CONTAINER_CLEAR_MM, "edge_mm": EDGE_MARGIN_MM,
                           "piece_sep_mm": MIN_SEP_MM, "arm_px": ARM_CLEAR_PX},
            "yaw": "degrees, anticlockwise in the table frame, uniform 0-360"}
    with open(f"{OUT}/layouts.json", "w") as f:
        json.dump({"meta": meta, "layouts": layouts}, f, indent=1)
    if args.preview:
        cv2.imwrite(f"{OUT}/allowed_area.jpg", preview(geo))
    n_eval = sum(L["set"] == "eval" for L in layouts)
    print(f"{n_eval} evaluation and {len(layouts) - n_eval} demonstration layouts -> {OUT}/")


if __name__ == "__main__":
    main()
