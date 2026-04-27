import csv
import json
import math
import random
from itertools import product
from pathlib import Path
from copy import deepcopy

# ---------------------- CONFIG ----------------------
SEED = 42
PARTICIPANTS = 12
TRIALS_PER_CONDITION = 1

# Grid / slots like spawn_train_layout
GRID_W, GRID_H = 13, 29
CAR_W, CAR_H = 2, 4
COL_ANCHORS = [1, 4, 7, 10]
ROW_ANCHORS = [0, 10, 15, 25]

# Colors (match env)
COLOR_IDS = [0, 1, 2]
COLOR_TO_PREFAB = {0: "black", 1: "red", 2: "white"}

# Condition factors
ANGLE_BUCKETS = {
    "front": (-55, 55),
    "side": ((55, 125), (-125, -55)),
    "back": (125, 180),
}
# NUM_CARS_LEVELS = [3, 5, 7]            # total cars INCLUDING target, per condition
NUM_CARS_LEVELS = [5, 7, 12]            # main study
# NUM_CARS_LEVELS = [1, 3, 6]           # training session
# VISUAL_DISTRACTOR_LEVELS = [0, 1, 2]   # training session
VISUAL_DISTRACTOR_LEVELS = [0, 2, 4]    # main study
# NOISE_LEVELS = {"none": {"background_noise": False}, "background": {"background_noise": True}}

PID_FMT = "p{:02d}"
DISCRETE_HEADINGS_DEG = [0, 90, 180, 270]

# Unity mapping (same as UnityProxyEnv._grid_to_unity)
X_MIN, X_MAX = 25.0, 38.0
Z_MIN, Z_MAX = -14.5, 14.5
# ----------------------------------------------------

random.seed(SEED)

# ---------------------- helpers ----------------------

def wrap_pi(theta_deg: float) -> float:
    return (theta_deg + 180.0) % 360.0 - 180.0

def in_bucket(rel_deg: float, bucket_key: str) -> bool:
    rel = wrap_pi(rel_deg)
    if bucket_key == "front":
        lo, hi = ANGLE_BUCKETS["front"]
        return lo <= rel <= hi
    if bucket_key == "side":
        lo, hi = random.choice(ANGLE_BUCKETS["side"])
        return lo <= rel <= hi
    if bucket_key == "back":
        lo, hi = ANGLE_BUCKETS["back"]
        return lo <= abs(rel) <= hi
    raise ValueError(bucket_key)

def footprint_cells(x0, y0):
    return [(x0 + dx, y0 + dy) for dx in range(CAR_W) for dy in range(CAR_H)]

def footprint_center(x0, y0):
    return x0 + CAR_W // 2, y0 + CAR_H // 2

def slot_index_to_center(idx: int):
    col = idx % len(COL_ANCHORS)
    row = idx // len(COL_ANCHORS)
    return footprint_center(COL_ANCHORS[col], ROW_ANCHORS[row])

def sample_slots(n_needed: int, rng: random.Random, exclude=None):
    slots = [(rx, ry) for ry in ROW_ANCHORS for rx in COL_ANCHORS]
    indices = list(range(len(slots)))
    rng.shuffle(indices)
    out, used = [], set(exclude or [])
    for idx in indices:
        if idx in used:
            continue
        out.append(idx)
        used.add(idx)
        if len(out) >= n_needed:
            break
    return out, used

def compute_rel_bearing_deg(agent_xy, agent_heading_deg, target_xy):
    ax, ay = agent_xy
    tx, ty = target_xy
    dx = tx - ax
    dy = ty - ay
    theta_abs_rad = math.atan2(dx, -dy) % (2 * math.pi)  # env's convention
    theta_abs_deg = math.degrees(theta_abs_rad)
    return wrap_pi(theta_abs_deg - agent_heading_deg)

def _expand_occupied_with_buffer(occupied_cells):
    """Return a set including all occupied cells and a 1-cell moat around them (Chebyshev distance <= 1)."""
    blocked = set(occupied_cells)
    for (cx, cy) in list(occupied_cells):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    blocked.add((nx, ny))
    return blocked

def choose_agent_pos_and_heading(target_center, occupied_cells, rng: random.Random, bucket_key: str):
    # Enforce >=1-cell clearance from ALL car footprint cells by expanding to a 1-cell moat
    blocked = _expand_occupied_with_buffer(occupied_cells)

    free = [
        (x, y)
        for x in range(GRID_W)
        for y in range(GRID_H)
        if (x, y) not in blocked
        and (x - target_center[0]) ** 2 + (y - target_center[1]) ** 2 >= 3 ** 2  # keep extra margin from target center
        and (x - target_center[0]) ** 2 + (y - target_center[1]) ** 2 <= GRID_W ** 2  # ensure agent is within GRID_W distance from target
    ]
    rng.shuffle(free)
    for (ax, ay) in free:
        heads = DISCRETE_HEADINGS_DEG[:]
        rng.shuffle(heads)
        for hdeg in heads:
            rel = compute_rel_bearing_deg((ax, ay), hdeg, target_center)
            if in_bucket(rel, bucket_key):
                return (ax, ay), hdeg
    # Fallback
    (ax, ay) = free[0] if free else (GRID_W // 2, GRID_H // 2)
    return (ax, ay), rng.choice(DISCRETE_HEADINGS_DEG)

def grid_to_unity(gx: int, gy: int):
    gw = max(1, GRID_W - 1)
    gh = max(1, GRID_H - 1)
    nx = float(gx) / gw
    ny = float(gy) / gh
    x = X_MIN + nx * (X_MAX - X_MIN)
    z = Z_MAX + ny * (Z_MIN - Z_MAX)
    return float(x), float(z)

# ---------------------- episode builder ----------------------

def make_episode(rng: random.Random, initial_angle_bucket: str, num_cars: int, visual_distractors: int):
    total_slots = len(COL_ANCHORS) * len(ROW_ANCHORS)  # 16
    if num_cars > total_slots:
        raise ValueError(f"num_cars={num_cars} exceeds available slots={total_slots}")
    if num_cars < 1:
        raise ValueError("num_cars must be >= 1 (needs at least a target)")

    # Target slot & color
    target_slot_idx = rng.randrange(total_slots)
    tcx, tcy = slot_index_to_center(target_slot_idx)
    target_color = rng.randrange(len(COLOR_IDS))

    used_slots = {target_slot_idx}

    # Number of non-target cars to place
    non_target_needed = num_cars - 1

    # Sample slots for non-target cars
    non_target_idxs, used_slots = sample_slots(non_target_needed, rng, exclude=used_slots)

    # Build object list (target first)
    objects = [(tcx, tcy, int(target_color))]

    # Assign colors to non-targets: exactly `visual_distractors` share the target color
    same_color_needed = max(0, min(int(visual_distractors), non_target_needed))
    for ni in non_target_idxs:
        nx, ny = slot_index_to_center(ni)
        if same_color_needed > 0:
            col = target_color
            same_color_needed -= 1
        else:
            other = [c for c in COLOR_IDS if c != target_color]
            col = rng.choice(other) if other else target_color
        objects.append((nx, ny, int(col)))

    # Occupied footprint cells
    occupied = set()
    all_used = {target_slot_idx, *non_target_idxs}
    for idx in all_used:
        col = idx % len(COL_ANCHORS)
        row = idx // len(COL_ANCHORS)
        x0 = COL_ANCHORS[col]; y0 = ROW_ANCHORS[row]
        for cell in footprint_cells(x0, y0):
            occupied.add(cell)

    # Agent placement
    (ax, ay), heading_deg = choose_agent_pos_and_heading((tcx, tcy), occupied, rng, initial_angle_bucket)

    # Output schema: vehicles/agent/background_noise/conditions
    vehicles = []
    for i, (cx, cy, col) in enumerate(objects):
        ux, uz = grid_to_unity(int(cx), int(cy))
        prefab = COLOR_TO_PREFAB.get(int(col), "black")
        vehicles.append({
            "name": f"{prefab.capitalize()}_Car_{i}",
            "position": {"x": ux, "y": 0.6, "z": uz},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            "isActive": True,
            "isTarget": (i == 0),
            "prefabType": prefab
        })

    ux_a, uz_a = grid_to_unity(int(ax), int(ay))
    agent = {
        "position": {"x": ux_a, "y": 0.0, "z": uz_a},
        "rotation": {"x": 0.0, "y": float(heading_deg), "z": 0.0},
    }

    return {
        "vehicles": vehicles,
        "agent": agent,
        # "background_noise": bool(NOISE_LEVELS[noise_key]["background_noise"]),
        "background_noise": True,  # Always on
    }

# ---------------------- pool & assignment ----------------------

def build_pool(seed=SEED):
    """Create the full pool of maps (episodes) once, with unique map_ids and conditions."""
    rng = random.Random(seed)

    angle_levels = ["front", "side", "back"]
    num_cars_levels = NUM_CARS_LEVELS
    distractor_levels = VISUAL_DISTRACTOR_LEVELS
    # noise_levels = list(NOISE_LEVELS.keys())

    # condition_grid = list(product(angle_levels, num_cars_levels, distractor_levels, noise_levels))
    condition_grid = list(product(angle_levels, num_cars_levels, distractor_levels))

    pool = []
    map_counter = 0
    for (angle, num_cars, distractors) in condition_grid:
        for _ in range(TRIALS_PER_CONDITION):
            ep = make_episode(rng, angle, num_cars, distractors)
            map_counter += 1
            map_id = f"M{map_counter:04d}"
            pool.append({
                "map_id": map_id,
                "conditions": {
                    "angle": angle,
                    "num_cars": int(num_cars),
                    "distractors": int(distractors),
                    # "noise": noise,
                },
                "episode": ep,
            })
    return pool


def assign_pool_to_participants(pool, n_participants=PARTICIPANTS, seed=SEED):
    """Return dict of participant->list(episodes) where each participant sees the same pool in a different random order.
    Also returns a list of (pid, order_index, map_id) rows for CSV.
    """
    rng = random.Random(seed)
    all_maps_by_id = {item["map_id"]: item for item in pool}

    assignments = {}
    csv_rows = []
    map_ids = [item["map_id"] for item in pool]

    for p in range(1, n_participants + 1):
        pid = PID_FMT.format(p)
        shuffled = map_ids[:]  # copy
        rng.shuffle(shuffled)

        # Build participant episode list by deep-copying the shared episodes and injecting map_id & conditions
        episodes = []
        for order_idx, mid in enumerate(shuffled):
            item = all_maps_by_id[mid]
            ep = deepcopy(item["episode"])  # copy to avoid mutating the pool
            # embed identifiers for traceability
            ep["map_id"] = mid
            ep["conditions"] = deepcopy(item["conditions"])
            episodes.append(ep)
            csv_rows.append({"participant": pid, "order_index": order_idx, "map_id": mid})

        assignments[pid] = episodes

    return assignments, csv_rows


def write_orders_csv(csv_rows, out_path: Path):
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["participant", "order_index", "map_id"])
        writer.writeheader()
        writer.writerows(csv_rows)

# ---------------------- main ----------------------

def generate_maps(n_participants=PARTICIPANTS, trials_per_condition=TRIALS_PER_CONDITION, seed=SEED):
    # Build the pool once (uses TRIALS_PER_CONDITION)
    pool = build_pool(seed=seed)

    # Assign pool to participants with per-participant randomization
    assignments, csv_rows = assign_pool_to_participants(pool, n_participants=n_participants, seed=seed)

    # Assemble the final maps.json structure:
    #  - top-level per-participant episodes (for backward compatibility)
    #  - include a top-level _pool index with map_id and conditions for auditability
    result = {pid: eps for pid, eps in assignments.items()}
    result["_pool"] = [{"map_id": item["map_id"], "conditions": item["conditions"]} for item in pool]

    return result, csv_rows

if __name__ == "__main__":
    maps, csv_rows = generate_maps(
        n_participants=PARTICIPANTS,
        trials_per_condition=TRIALS_PER_CONDITION,
        seed=SEED,
    )

    out_path = Path("maps_training.json")
    with out_path.open("w") as f:
        json.dump(maps, f, indent=2)
    print(f"Wrote {out_path.resolve()}")

    csv_path = Path("training_map_orders.csv")
    write_orders_csv(csv_rows, csv_path)
    print(f"Wrote {csv_path.resolve()}")