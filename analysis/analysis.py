"""Preregistered analysis for 'Who should move the arm?'.

Usage: python analysis/analysis.py results/trials.csv
Uses blind scores; standard library only.
"""
import csv
import math
import sys
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant pairs b (first only) and c (second only)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["scorer"] == "blind"]
    for r in rows:
        r["success"] = int(r["success"])
        r["grasp"] = int(int(r["progress"]) >= 2)
    return rows


def rate(rows, arm, key, conds):
    sel = [r[key] for r in rows if r["arm"] == arm and r["condition"] in conds]
    k, n = sum(sel), len(sel)
    lo, hi = wilson(k, n)
    return k, n, lo, hi


def paired(rows, a1, a2, key, conds):
    by = defaultdict(dict)
    for r in rows:
        if r["condition"] in conds:
            by[r["layout_id"]][r["arm"]] = r[key]
    pairs = [(v[a1], v[a2]) for v in by.values() if a1 in v and a2 in v]
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    return len(pairs), b, c, mcnemar_exact(b, c)


def verdict(diff, support, refute_below, refute_at=False):
    if diff >= support:
        return "supported"
    if diff < refute_below or (refute_at and diff == refute_below):
        return "refuted"
    return "inconclusive"


def main(path):
    rows = load(path)
    allc, off = {"E1", "E2", "E3"}, {"E2", "E3"}

    print("Success by arm and condition (Wilson 95%)")
    for arm in ["A", "B", "C", "C-ACT"]:
        for cond in ["E1", "E2", "E3"]:
            k, n, lo, hi = rate(rows, arm, "success", {cond})
            print(f"  {arm:6} {cond}: {k}/{n}  [{lo:.2f}, {hi:.2f}]")

    tests = [
        ("H1", "C", "B", "success", off, 25, 10, False),
        ("H2", "C", "A", "grasp", allc, 20, 0, True),
        ("H4", "C-ACT", "C", "success", allc, -15, -15, False),
    ]
    for name, x, y, key, conds, sup, ref, at in tests:
        kx, nx, _, _ = rate(rows, x, key, conds)
        ky, ny, _, _ = rate(rows, y, key, conds)
        diff = 100 * (kx / nx - ky / ny) if nx and ny else float("nan")
        n, b, c, p = paired(rows, x, y, key, conds)
        tie = " (tie: |b-c| <= 2)" if abs(b - c) <= 2 else ""
        print(f"{name}: {x} - {y} = {diff:+.0f} pts, pairs={n}, discordant {b}/{c}, "
              f"McNemar p={p:.3f}, {verdict(diff, sup, ref, at)}{tie}")

    k, n, lo, hi = rate(rows, "A", "success", {"E1"})
    pct = 100 * k / n if n else float("nan")
    print(f"H3: A on E1 = {pct:.0f}% [{lo:.2f}, {hi:.2f}], {verdict(pct, 30, 15)}")


if __name__ == "__main__":
    main(sys.argv[1])
