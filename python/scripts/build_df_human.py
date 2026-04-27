#!/usr/bin/env python3
"""Rebuild df_human-style CSV directly from human_root without filtering outliers.

The script walks each participant folder under --human-root, looks up map_id and
conditions from maps.json, computes simple trajectory metrics from the
agent_data_*.json files, and writes a CSV compatible with analysis/df_human.csv.
"""
import argparse
import csv
import glob
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse Unity timestamp strings with or without fractional seconds."""
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            continue
    return None


def _load_scene(ep_dir: str) -> Dict[str, Any]:
    """Load the first scene_data_*.json if present."""
    paths = sorted(glob.glob(os.path.join(ep_dir, "scene_data_*.json")))
    if not paths:
        return {}
    try:
        with open(paths[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load scene data in {ep_dir}: {e}")
        return {}


def _load_estimate(ep_dir: str) -> Optional[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(ep_dir, "estimate*.json")))
    if not paths:
        return None
    try:
        with open(paths[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load estimate in {ep_dir}: {e}")
        return None


def _load_frames(ep_dir: str) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for fp in sorted(glob.glob(os.path.join(ep_dir, "agent_data_*.json"))):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] failed to read frame {fp}: {e}")
            continue
        frames.append(
            {
                "timestamp": _parse_timestamp(data.get("timestamp")),
                "step": data.get("step"),
                "position": data.get("position"),
                "rotation": data.get("rotation"),
            }
        )
    frames.sort(
        key=lambda fr: (
            1 if fr["timestamp"] is None else 0,
            fr["timestamp"] or datetime.min,
            fr["step"] if isinstance(fr["step"], (int, float)) else 0,
        )
    )
    return frames


def _angle_diff(a: float, b: float) -> float:
    """Return wrapped difference b-a in degrees."""
    diff = b - a
    while diff > 180.0:
        diff -= 360.0
    while diff < -180.0:
        diff += 360.0
    return diff


def _compute_metrics(
    frames: List[Dict[str, Any]],
    vehicles: Iterable[Dict[str, Any]],
    estimate: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Mimic compare_traj.compute_metrics but without any filtering."""
    if not frames:
        return None

    start = frames[0].get("position") or {}
    end = frames[-1].get("position") or {}
    if not start or not end:
        return None

    sx, sz = float(start.get("x", 0.0)), float(start.get("z", 0.0))
    ex, ez = float(end.get("x", 0.0)), float(end.get("z", 0.0))
    straight_dist = math.hypot(ex - sx, ez - sz)

    t0, t1 = frames[0].get("timestamp"), frames[-1].get("timestamp")
    dur = None
    if t0 and t1:
        dur = (t1 - t0).total_seconds()
    else:
        s0, s1 = frames[0].get("step"), frames[-1].get("step")
        if isinstance(s0, (int, float)) and isinstance(s1, (int, float)):
            dur = (float(s1) - float(s0)) / 1.31  # mirror compare_traj heuristic
    speed = straight_dist / dur if (dur and dur > 0) else 0.0

    acc_turns = 0.0
    for a, b in zip(frames, frames[1:]):
        ra = (a.get("rotation") or {}).get("y")
        rb = (b.get("rotation") or {}).get("y")
        if ra is None or rb is None:
            continue
        acc_turns += abs(_angle_diff(float(ra), float(rb)))

    target_pos = None
    for v in vehicles or []:
        try:
            if v.get("isTarget") or "Target" in v.get("name", ""):
                pos = v.get("position", {})
                if "x" in pos and "z" in pos:
                    target_pos = pos
                    break
        except Exception:
            continue

    end_to_target = None
    if target_pos:
        end_to_target = math.hypot(ex - float(target_pos["x"]), ez - float(target_pos["z"]))

    estimate_to_target = None
    estimate_correct = None
    if estimate and target_pos and ("x" in estimate) and ("z" in estimate):
        estimate_to_target = math.hypot(
            float(estimate["x"]) - float(target_pos["x"]),
            float(estimate["z"]) - float(target_pos["z"]),
        )
        closest = None
        min_d = float("inf")
        for v in vehicles or []:
            pos = v.get("position", {})
            if "x" not in pos or "z" not in pos:
                continue
            d = math.hypot(float(estimate["x"]) - float(pos["x"]), float(estimate["z"]) - float(pos["z"]))
            if d < min_d:
                min_d = d
                closest = v
        if closest is not None:
            estimate_correct = bool(closest.get("isTarget") or "Target" in closest.get("name", ""))

    return {
        "distance": straight_dist,
        "duration": dur,
        "speed": speed,
        "accumulated_head_turns": acc_turns,
        "end_x": ex,
        "end_z": ez,
        "end_to_target_dist": end_to_target,
        "estimate_correct": 1 if estimate_correct else 0,
        "estimate_to_target_dist": estimate_to_target,
    }


def build_rows(human_root: str, maps_json: str) -> List[Dict[str, Any]]:
    with open(maps_json, "r", encoding="utf-8") as f:
        maps_data = json.load(f)

    rows: List[Dict[str, Any]] = []
    for pid in sorted(k for k in maps_data.keys() if k.startswith("p")):
        maps_list = maps_data.get(pid) or []
        for idx, entry in enumerate(maps_list):
            ep_dir = os.path.join(human_root, pid, str(idx))
            if not os.path.isdir(ep_dir):
                print(f"[warn] missing episode directory: {ep_dir}")
                continue
            frames = _load_frames(ep_dir)
            scene = _load_scene(ep_dir)
            est = _load_estimate(ep_dir)
            metrics = _compute_metrics(frames, scene.get("vehicles", []), est)
            if metrics is None:
                print(f"[warn] could not compute metrics for {pid} idx {idx}; row will be skipped")
                continue

            cond = entry.get("conditions", entry.get("condition", {})) or {}
            row = {
                "map_id": entry.get("map_id", entry.get("id")),
                "participant": pid,
                "agent_type": "human",
                "distance": metrics["distance"],
                "duration": metrics["duration"],
                "speed": metrics["speed"],
                "angle": cond.get("angle"),
                "num_cars": cond.get("num_cars"),
                "distractors": cond.get("distractors"),
                "accumulated_head_turns": metrics["accumulated_head_turns"],
                "end_x": metrics["end_x"],
                "end_z": metrics["end_z"],
                "end_to_target_dist": metrics["end_to_target_dist"],
                "estimate_correct": metrics["estimate_correct"],
                "estimate_to_target_dist": metrics["estimate_to_target_dist"],
                "idx": idx,
                "actor": "human",
            }
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Recreate df_human.csv directly from human_root logs.")
    ap.add_argument("--human-root", default="analysis/data", help="Root folder containing p01...p12")
    ap.add_argument("--maps-json", default="maps/maps.json", help="maps.json with map_id/condition entries")
    ap.add_argument("--out", default="analysis/data/df_human.csv", help="Where to write the CSV")
    args = ap.parse_args()

    rows = build_rows(args.human_root, args.maps_json)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "map_id",
        "participant",
        "agent_type",
        "distance",
        "duration",
        "speed",
        "angle",
        "num_cars",
        "distractors",
        "accumulated_head_turns",
        "end_x",
        "end_z",
        "end_to_target_dist",
        "estimate_correct",
        "estimate_to_target_dist",
        "idx",
        "actor",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[ok] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
