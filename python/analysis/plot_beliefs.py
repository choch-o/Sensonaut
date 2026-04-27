#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_beliefs.py
---------------
Visualize belief / likelihood maps from a single participant & trial.

Outputs:
- plots/beliefs_{pid}_{trial}.png            (default: last step)
- plots/beliefs_{pid}_{trial}_stepXXXX.png   (if --all-steps or --step given)

Usage:
  python plot_beliefs.py --pid agent_3 --trial 12
  python plot_beliefs.py --pid agent_3 --trial 12 --step 25
  python plot_beliefs.py --pid agent_3 --trial 12 --all-steps
"""


import os
import sys
import glob
import json
import pickle
import argparse
import numpy as np

# Ensure parent directory (repo root) is importable when running this file directly
_CUR_DIR = os.path.dirname(__file__)
_PARENT_DIR = os.path.abspath(os.path.join(_CUR_DIR, os.pardir))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
import matplotlib.pyplot as plt
from compare_traj import draw_episode_like_maps, plot_trajectory, load_scene_data, unity_to_grid
from utils.constants import THETA_GRID, R_GRID

# Set Roboto as the default sans-serif font
# plt.rcParams['font.sans-serif'] = ['Helvetica'] 
# plt.rcParams['font.family'] = 'sans-serif' 

TITLE_SIZE = 12

def load_agent_data(file_path):
    """
    Load a single frame file.
    - Human runs typically store JSON (agent_*.json) with timestamps and no arrays.
    - Agent runs typically store pickle (agent_*.pkl) with arrays (prior, audio_like, ...)
    """
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(file_path, "rb") as f:
        return pickle.load(f)

def get_agent_trajectory(trial_dir):
    """
    Load all agent_* frames from the given trial directory as a list (sorted).
    Returns a list of dicts. For agent (pickle) frames, the last step usually has
    'prior', 'audio_like', 'visual_like', 'log_visual', 'log_audio', 'log_posterior'.
    """
    if "agent_" in trial_dir:
        # still safe, we match by extensions below
        pass
    # Pick up both formats
    pkl_files = glob.glob(os.path.join(trial_dir, "agent_*.pkl"))
    json_files = glob.glob(os.path.join(trial_dir, "agent_*.json"))
    files = sorted(pkl_files + json_files)
    traj = []
    for fp in files:
        try:
            traj.append(load_agent_data(fp))
        except Exception as e:
            print(f"[warn] Failed to load {fp}: {e}")
    return traj

def _imshow_panel(ax, arr, title, cmap):
    if arr is None:
        ax.set_axis_off()
        return
    im = ax.imshow(arr, cmap=cmap, origin='lower', aspect='auto')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def _overlay_belief_on_world(
    ax,
    step_dict,
    arr,
    cmap='Greens',
    alpha=0.2,
    blur_sigma=(0.0, 0.0),   # (sigma_r, sigma_theta) in *bins*, e.g., (0.7, 1.5)
    upsample=1,              # integer upsample factor, e.g., 2–4
    point_size=8,            # scatter size for the smaller cells
    agent_color='red',
    norm=None,
    draw_colorbar=False
):
    """
    Overlay an egocentric (r x theta) belief array on the world map by projecting
    it around the agent. Optionally smooth (Gaussian on r/theta) and upsample to
    reduce the 'radar rings' / blockiness from discrete bins.
    """
    if arr is None:
        return

    try:
        # --- set up original axes (r, theta) ---
        thetas0 = THETA_GRID
        rs0 = R_GRID
        arr = np.asarray(arr, dtype=float)

        if arr.shape != (len(rs0), len(thetas0)):
            print(f"[warn] belief array shape mismatch; expected {(len(rs0), len(thetas0))}, got {arr.shape}")
            # Fallback if shapes differ from constants
            thetas0 = np.linspace(-np.pi, np.pi, arr.shape[1])
            r_min = float(rs0.min()) if hasattr(R_GRID, 'min') else 0.5
            r_max = float(rs0.max()) if hasattr(R_GRID, 'max') else 30.0
            rs0 = np.linspace(r_min, r_max, arr.shape[0])

        # --- normalize to [0,1] (handles log inputs) ---
        if not np.any(np.isfinite(arr)):
            return
        if np.nanmax(arr) > 0 and np.nanmin(arr) < 0:
            arr = np.exp(arr - np.nanmax(arr))  # log → prob-like
        else:
            denom = (np.nanmax(arr) - np.nanmin(arr) + 1e-12)
            arr = (arr - np.nanmin(arr)) / denom
        # arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # --- optional Gaussian blur in (r, theta) space ---
        sig_r, sig_th = blur_sigma if isinstance(blur_sigma, (tuple, list)) else (blur_sigma, blur_sigma)
        if (sig_r and sig_r > 0) or (sig_th and sig_th > 0):
            try:
                # SciPy path (best)
                import scipy.ndimage as ndi
                arr = ndi.gaussian_filter(arr, sigma=(sig_r, sig_th), mode='nearest')
            except Exception:
                # NumPy fallback: separable Gaussian
                def _gauss_1d(sigma):
                    w = max(3, int(4.0 * sigma + 1))  # ~±2σ support
                    xs = np.arange(-w, w + 1)
                    k = np.exp(-0.5 * (xs / (sigma + 1e-12))**2)
                    return k / (k.sum() + 1e-12)

                def _conv1d_along_axis(x, k, axis):
                    # pad edge values (nearest)
                    pad = (len(k) - 1) // 2
                    xpad = np.take(x, [0], axis=axis)
                    xpre = np.repeat(xpad, pad, axis=axis)
                    xpad = np.take(x, [-1], axis=axis)
                    xpost = np.repeat(xpad, pad, axis=axis)
                    xcat = np.concatenate([xpre, x, xpost], axis=axis)
                    # roll-conv
                    out = np.zeros_like(x, dtype=float)
                    for i, w in enumerate(k):
                        shift = i - pad
                        out += w * np.take(xcat, range(pad + shift, pad + shift + x.shape[axis]), axis=axis)
                    return out

                if sig_r and sig_r > 0:
                    kr = _gauss_1d(sig_r)
                    arr = _conv1d_along_axis(arr, kr, axis=0)
                if sig_th and sig_th > 0:
                    kth = _gauss_1d(sig_th)
                    arr = _conv1d_along_axis(arr, kth, axis=1)

        # --- optional upsample in (r, theta) space (bilinear-ish) ---
        if upsample and upsample > 1:
            new_n_r = arr.shape[0] * upsample
            new_n_th = arr.shape[1] * upsample
            rs = np.linspace(rs0[0], rs0[-1], new_n_r)
            thetas = np.linspace(thetas0[0], thetas0[-1], new_n_th)

            # separable linear interpolation without SciPy
            # first along theta
            th_old = thetas0
            th_new = thetas
            arr_th = np.empty((arr.shape[0], new_n_th), dtype=float)
            for i in range(arr.shape[0]):
                arr_th[i, :] = np.interp(th_new, th_old, arr[i, :])
            # then along r
            r_old = rs0
            r_new = rs
            arr_up = np.empty((new_n_r, new_n_th), dtype=float)
            for j in range(arr_th.shape[1]):
                arr_up[:, j] = np.interp(r_new, r_old, arr_th[:, j])
            arr = arr_up
            rs0, thetas0 = rs, thetas  # replace with dense axes

        # --- project to world coordinates ---
        ax_u = step_dict['position']['x']
        az_u = step_dict['position']['z']
        gx_a, gy_a = unity_to_grid(ax_u, az_u)
        heading = np.deg2rad(step_dict['rotation'].get('y', 0.0))

        dx, dy = 2.0*np.sin(heading), -2.0*np.cos(heading)  # y-axis flip
        ax.plot([gx_a], [gy_a], marker='o', markersize=3, color=agent_color, zorder=5)
        ax.arrow(gx_a, gy_a, dx, dy, head_width=0.45, head_length=0.6,
                length_includes_head=True, color=agent_color, zorder=5, linewidth=1.0)

        theta_grid, r_grid = np.meshgrid(thetas0, rs0, indexing='xy')
        gx = gx_a + r_grid * np.sin(heading + theta_grid)
        gy = gy_a - r_grid * np.cos(heading + theta_grid)  # y-axis flip

        # --- draw dense + smooth heatmap (scatter-of-squares) ---
        # vmin = 0.0
        # vmax = 0.5
        ax.scatter(
            gx, gy,
            c=arr,
            cmap=cmap,
            # vmin=vmin,
            # vmax=vmax,
            s=point_size,
            marker='s',
            alpha=alpha,
            linewidths=0,
            edgecolors='none'
        )

        # colorbar (kept simple)
        mappable = plt.cm.ScalarMappable(cmap=cmap) #, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        mappable.set_array(arr)
        if draw_colorbar:
            plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.01)

    except Exception as e:
        ax.text(0.02, 0.02, f"(overlay failed)\n{e}", transform=ax.transAxes,
                ha='left', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

def save_belief_grid_for_step(step_dict, out_path, scene_data=None, trajectory=None, step_idx=None, pid_label=""):
    """
    Save a 1x4 grid with:
      [0] world grid (scene) + trajectory up to step_idx
      [1] log_visual
      [2] log_audio
      [3] log_posterior

    Expects belief/likelihood keys in step_dict. If scene_data or trajectory
    are missing, the leftmost panel will be blank.
    """
    if step_dict is None or not isinstance(step_dict, dict):
        print("[skip] step_dict is empty or not a dict.")
        return

    # Pull arrays if present
    prior         = step_dict.get("prior")
    audio_like    = step_dict.get("audio_like")
    visual_like   = step_dict.get("visual_like")
    log_visual    = step_dict.get("log_visual")
    log_audio     = step_dict.get("log_audio")
    log_posterior = step_dict.get("log_posterior")

    if not any(x is not None for x in [prior, audio_like, visual_like, log_visual, log_audio, log_posterior]):
        print("[skip] No belief/likelihood arrays found in this step.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Figure: 1 row, 4 columns (world + three heatmaps)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)

    # [0] World grid + (partial) trajectory (NO overlay)
    ax_world = axes[0]
    if scene_data is not None:
        try:
            draw_episode_like_maps(
                ax_world,
                scene_data,
                show_grid=False,
                tiny=False,
                agent_override=step_dict,
            )
            if trajectory:
                # Clamp step index
                if isinstance(step_idx, int):
                    k = max(0, min(step_idx, len(trajectory)-1))
                    partial_traj = trajectory[:k+1]
                else:
                    partial_traj = trajectory
                plot_trajectory(ax_world, partial_traj, pid_label or "traj", color=None,
                                show_rotation=False, show_fov=False, maps_style=True)
            ax_world.set_title("World", fontsize=TITLE_SIZE)
        except Exception as e:
            ax_world.set_axis_off()
            ax_world.text(0.5, 0.5, f"(world plot failed)\n{e}", ha="center", va="center", fontsize=8)
    else:
        ax_world.set_axis_off()
        ax_world.text(0.5, 0.5, "(no scene_data)", ha="center", va="center", fontsize=8)

    # [1] World + log_visual overlay
    ax_vis = axes[1]
    if scene_data is not None:
        draw_episode_like_maps(ax_vis, scene_data, show_grid=False, tiny=False, agent_override=step_dict)
        _overlay_belief_on_world(ax_vis, step_dict, log_visual, cmap='Blues', alpha=0.5)
        ax_vis.set_title("World + Log Visual", fontsize=TITLE_SIZE)
    else:
        _imshow_panel(ax_vis, log_visual, "log_visual", "Blues")

    # [2] World + log_audio overlay
    ax_aud = axes[2]
    if scene_data is not None:
        draw_episode_like_maps(ax_aud, scene_data, show_grid=False, tiny=False, agent_override=step_dict)
        _overlay_belief_on_world(ax_aud, step_dict, log_audio, cmap='Oranges', alpha=0.5)
        ax_aud.set_title("World + Log Audio", fontsize=TITLE_SIZE)
    else:
        _imshow_panel(ax_aud, log_audio, "log_audio", "Oranges")

    # [3] World + log_posterior overlay
    ax_post = axes[3]
    if scene_data is not None:
        draw_episode_like_maps(ax_post, scene_data, show_grid=False, tiny=False, agent_override=step_dict)
        _overlay_belief_on_world(ax_post, step_dict, log_posterior, cmap='viridis', alpha=0.5)
        ax_post.set_title("World + Log Posterior", fontsize=TITLE_SIZE)
    else:
        _imshow_panel(ax_post, log_posterior, "log_posterior", "viridis")

    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot belief/likelihood maps for a single participant trial.")
    parser.add_argument("--pid", required=True, type=str, help="Participant ID (e.g., p01, agent_3)")
    parser.add_argument("--model", required=True, type=str, help="Model name (e.g., agent-turn90-step1)")
    parser.add_argument("--trial", required=True, type=str, help="Trial dir under ./data/{pid}/ (e.g., 5)")
    parser.add_argument("--outdir", type=str, default="belief_plots", help="Output directory (default: plots)")
    parser.add_argument("--step", type=str, default="last",
                        help='Step index to export (int) or "last" (default: last)')
    parser.add_argument("--all-steps", action="store_true", help="Export all steps")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI (default: 200)")
    args = parser.parse_args()

    base_dir = os.path.join("../test_logs/", args.model, args.pid, str(args.trial))
    if not os.path.isdir(base_dir):
        print(f"[error] Directory not found: {base_dir}")
        return

    # Load scene data once for this trial
    scene_data = load_scene_data(base_dir)

    traj = get_agent_trajectory(base_dir)
    if not traj:
        print(f"[error] No agent_* frames found in {base_dir}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    def save_for_idx(idx):
        if idx < 0 or idx >= len(traj):
            print(f"[warn] step {idx} out of range (0..{len(traj)-1}). Skipping.")
            return
        step_dict = traj[idx] if isinstance(traj[idx], dict) else None
        if step_dict is None:
            print(f"[warn] step {idx} is not a dict (type={type(traj[idx])}). Skipping.")
            return
        out_path = os.path.join(args.outdir, f"beliefs_{args.pid}_{args.trial}_step{idx:04d}.png")
        save_belief_grid_for_step(step_dict, out_path, scene_data=scene_data, trajectory=traj, step_idx=idx, pid_label=args.pid)
        print(f"[ok] Saved {out_path}")

    if args.all_steps:
        for i in range(len(traj)):
            save_for_idx(i)
        return

    if args.step == "last":
        # last step
        idx = len(traj) - 1
        out_path = os.path.join(args.outdir, f"beliefs_{args.pid}_{args.trial}.png")
        step_dict = traj[idx] if isinstance(traj[idx], dict) else None
        if step_dict is None:
            print("[error] Last step is not a dict; cannot extract belief arrays.")
            return
        save_belief_grid_for_step(step_dict, out_path, scene_data=scene_data, trajectory=traj, step_idx=idx, pid_label=args.pid)
        print(f"[ok] Saved {out_path}")
    else:
        try:
            idx = int(args.step)
        except ValueError:
            print(f"[error] --step must be an integer or 'last', got: {args.step}")
            return
        save_for_idx(idx)

if __name__ == "__main__":
    main()
