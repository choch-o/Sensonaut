# visualize_maps.py
# Modes:
#  (A) Default: per-participant grids (6x9) -> same as before
#  (B) --by-condition: make ONE plot per unique condition, showing all episodes
#      - by default, aggregates episodes across ALL selected participants
#      - use --per-participant to produce separate figures per participant per condition

import json
import math
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Constants (must mirror your env) ---
GRID_W, GRID_H = 13, 29       # logical grid
CAR_W, CAR_H   = 2, 4         # car footprint (cells)

# UnityProxyEnv._grid_to_unity() bounds
X_MIN, X_MAX = 25.0, 38.0
Z_MIN, Z_MAX = -14.5, 14.5

# Colors for prefabs
COLOR_MAP = {
    "black":  "#2e2e2e",
    "red":    "#4a9c9c",
    "white":  "#dddddd",
}

# ----------------- shared helpers -----------------

def unity_to_grid(ux: float, uz: float):
    """
    Inverse of _grid_to_unity:
      x = X_MIN + (gx/gw)*(X_MAX - X_MIN)
      z = Z_MAX + (gy/gh)*(Z_MIN - Z_MAX)
    """
    gw = max(1, GRID_W - 1)
    gh = max(1, GRID_H - 1)
    nx = (ux - X_MIN) / (X_MAX - X_MIN)
    ny = (uz - Z_MAX) / (Z_MIN - Z_MAX)  # note reversed order vs typical
    gx = int(round(nx * gw))
    gy = int(round(ny * gh))
    # Clamp just in case of tiny FP drift
    gx = max(0, min(GRID_W - 1, gx))
    gy = max(0, min(GRID_H - 1, gy))
    return gx, gy


def draw_episode(ax, episode, show_grid=False, tiny=False, label=None):
    """Draw one episode on a provided Axes."""
    # Optional minimalist grid
    if show_grid:
        ax.set_xticks(range(GRID_W), minor=False)
        ax.set_yticks(range(GRID_H), minor=False)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.25)
    else:
        ax.grid(False)

    # Keep aspect and flip y so (0,0) is bottom-left
    ax.set_aspect('equal')
    ax.set_xlim(0, X_MAX-X_MIN)
    ax.set_ylim(Z_MAX-Z_MIN, 0)  # reverse y-axis
    ax.set_xticks([])
    ax.set_yticks([])

    # Cars
    for v in sorted(episode["vehicles"], key=lambda d: d.get("isTarget", False)):
        ux, uz = v["position"]["x"], v["position"]["z"]
        gx, gy = unity_to_grid(ux, uz)
        prefab = v.get("prefabType", "black")
        is_target = bool(v.get("isTarget", False))

        face = COLOR_MAP.get(prefab, "#888888")
        x0 = gx - CAR_W / 2.0
        y0 = gy - CAR_H / 2.0
        rect = patches.Rectangle(
            (x0, y0), CAR_W, CAR_H,
            linewidth=1.6 if is_target else 0.8,
            edgecolor="#ff00aa" if is_target else "#000000",
            facecolor=face,
            alpha=0.82 if is_target else 0.6
        )
        ax.add_patch(rect)

    # Agent
    ax_u = episode["agent"]["position"]["x"]
    az_u = episode["agent"]["position"]["z"]
    gx_a, gy_a = unity_to_grid(ax_u, az_u)
    heading_deg = float(episode["agent"]["rotation"]["y"])
    ax.plot([gx_a], [gy_a], marker="o", markersize=2.5 if tiny else 4, color="#111111")
    # Heading: 0° = up/north
    L = 2.0 if tiny else 2.5
    th = math.radians(heading_deg)
    dx =  L * math.sin(th)
    dy = -L * math.cos(th)
    ax.arrow(gx_a, gy_a, dx, dy, head_width=0.45 if tiny else 0.6,
             head_length=0.7 if tiny else 0.9, length_includes_head=True, color="#111111", linewidth=0.8)

    # Corner label (index, cond snippet, etc.)
    if label is not None:
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=6 if tiny else 7, color="#222",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.3))

# ----------------- per-participant grid (existing) -----------------

def make_grid_for_participant(pid: str, episodes, cols=6, rows=9, start=0, show_grid=False, save_dir=Path("viz_grids")):
    """Render cols×rows (default 6×9=54) episodes for one participant."""
    total = cols * rows
    subset = episodes[start:start+total]
    save_dir.mkdir(parents=True, exist_ok=True)

    fig_w = cols * 2.2
    fig_h = rows * 2.8
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.flatten()

    for i in range(total):
        ax = axes[i]
        if i < len(subset):
            ep = subset[i]
            label = f"{pid} #{start+i} {ep['map_id']}"
            c = ep.get("conditions", {})
            if c:
                label += f"\n{c.get('angle','')},#cars={c.get('num_cars','')},dist={c.get('distractors','')},{'bg' if ep.get('background_noise') else 'no-bg'}"
            draw_episode(ax, ep, show_grid=show_grid, tiny=True, label=label)
        else:
            ax.axis("off")

    fig.suptitle(f"{pid} — {len(subset)} episodes (from {start})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = save_dir / f"{pid}_grid_{cols}x{rows}_start{start}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out.resolve()}")

# ----------------- group-by-condition mode -----------------

def condition_key(ep):
    c = ep.get("conditions", {})
    # Normalize keys used by your generator
    return (
        c.get("angle", ""),
        int(c.get("num_cars", -1)) if str(c.get("num_cars", "")).strip() != "" else -1,
        int(c.get("distractors", -1)) if str(c.get("distractors", "")).strip() != "" else -1,
        # c.get("noise", "")
    )


def key_to_str(key_tuple):
    angle, num_cars, distractors = key_tuple
    # noise_tag = "bg" if noise in ("background", True, "true", "True") else ("no-bg" if noise in ("none", False, "false", "False") else str(noise))
    return f"angle={angle} | num_cars={num_cars} | distractors={distractors}" # | noise={noise_tag}"


def auto_grid(n):
    if n <= 0:
        return 1, 1
    # favor wider-than-tall a bit for readability
    cols = min(10, max(3, int(round(math.sqrt(n)*1.2))))
    rows = int(math.ceil(n / cols))
    return cols, rows


def make_condition_fig(episodes, cond_key, title_prefix="ALL" , show_grid=False, save_dir=Path("viz_by_condition")):
    """Make one figure for a given condition key with all provided episodes."""
    n = len(episodes)
    if n == 0:
        return None
    save_dir.mkdir(parents=True, exist_ok=True)

    cols, rows = auto_grid(n)
    fig_w = cols * 2.1
    fig_h = rows * 2.6
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.flatten() if isinstance(axes, (list, tuple)) else axes.ravel()

    for i in range(rows * cols):
        ax = axes[i]
        if i < n:
            ep = episodes[i]
            pid_label = ep.get("_pid", "")
            idx_label = ep.get("_idx", None)
            lbl = f"{pid_label}#{idx_label}" if idx_label is not None else pid_label
            draw_episode(ax, ep, show_grid=show_grid, tiny=True, label=lbl)
        else:
            ax.axis("off")

    cond_str = key_to_str(cond_key)
    fig.suptitle(f"{title_prefix} — {cond_str}  (n={n})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    angle, num_cars, distractors = cond_key
    # noise_tag = "bg" if noise == "background" else ("no-bg" if noise == "none" else str(noise))
    fname = f"{title_prefix.replace(' ','_')}_angle-{angle}_cars-{num_cars}_dist-{distractors}.png"
    out = save_dir / fname
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Saved {out.resolve()}")
    return out

# ----------------- CLI -----------------

def main():
    parser = argparse.ArgumentParser(description="Visualize maps as grids or grouped by condition.")
    parser.add_argument("--maps", type=str, default="maps.json", help="Path to maps.json")

    # Existing per-participant grid options
    parser.add_argument("--cols", type=int, default=6, help="Columns for per-participant grid (default 6)")
    parser.add_argument("--rows", type=int, default=9, help="Rows for per-participant grid (default 9)")
    parser.add_argument("--start", type=int, default=0, help="Start index within each participant's episodes")
    parser.add_argument("--participants", type=str, nargs="*", default=None, help="Subset of pIDs (e.g., p01 p02). Default: all.")
    parser.add_argument("--gridlines", action="store_true", help="Show gridlines in each tile")
    parser.add_argument("--outdir", type=str, default="viz_grids", help="Output directory for per-participant images")

    # Group-by-condition mode
    parser.add_argument("--by-condition", action="store_true", help="Group episodes by unique condition and make one plot per condition")
    parser.add_argument("--per-participant", action="store_true", help="When grouping by condition, make a separate figure per participant per condition")
    parser.add_argument("--cond-outdir", type=str, default="viz_by_condition", help="Output directory for condition figures")

    args = parser.parse_args()

    path = Path(args.maps)
    with path.open("r") as f:
        maps = json.load(f)

    # Participant selection
    pids = sorted(maps.keys())
    if args.participants:
        requested = set(args.participants)
        pids = [p for p in pids if p in requested]
        if not pids:
            raise SystemExit(f"No matching participants among {sorted(maps.keys())}")

    if not args.by_condition:
        # ---- Mode A: per-participant fixed grids ----
        for pid in pids:
            make_grid_for_participant(
                pid,
                maps[pid],
                cols=args.cols,
                rows=args.rows,
                start=args.start,
                show_grid=args.gridlines,
                save_dir=Path(args.outdir),
            )
        return

    # ---- Mode B: group-by-condition ----
    if args.per_participant:
        # separate figure per participant per condition
        for pid in pids:
            eps = maps[pid]
            # attach metadata for labels
            for i, ep in enumerate(eps):
                ep["_pid"], ep["_idx"] = pid, i
            # group
            groups = {}
            for i, ep in enumerate(eps):
                k = condition_key(ep)
                groups.setdefault(k, []).append(ep)
            # render
            for k, lst in sorted(groups.items()):
                make_condition_fig(lst, k, title_prefix=pid, show_grid=args.gridlines, save_dir=Path(args.cond_outdir))
    else:
        # aggregate across selected participants
        groups = {}
        for pid in pids:
            for i, ep in enumerate(maps[pid]):
                ep["_pid"], ep["_idx"] = pid, i
                k = condition_key(ep)
                groups.setdefault(k, []).append(ep)
        for k, lst in sorted(groups.items()):
            make_condition_fig(lst, k, title_prefix="ALL", show_grid=args.gridlines, save_dir=Path(args.cond_outdir))

if __name__ == "__main__":
    main()