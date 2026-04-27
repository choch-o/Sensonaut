import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.signal import resample
from datetime import datetime
import argparse
import pickle
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# Add parent directory for utils imports
_CUR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import shared constants and utilities
from utils.constants import (
    GRID_W, GRID_H, CAR_W, CAR_H,
    X_MIN, X_MAX, Z_MIN, Z_MAX,
    COLOR_MAP,
)
from utils.coordinates import unity_to_grid

def draw_episode_like_maps(ax, episode, show_grid=False, tiny=False, label=None, agent_override=None, draw_agent=True):
    """Mirror visualize_maps.draw_episode: same cars, agent arrow, axes style."""
    import math
    # Optional minimalist grid
    if show_grid:
        ax.set_xticks(range(GRID_W), minor=False)
        ax.set_yticks(range(GRID_H), minor=False)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.25)
    else:
        ax.grid(False)

    # Keep aspect and flip y so (0,0) is bottom-left
    ax.set_aspect('equal')
    ax.set_xlim(0, X_MAX - X_MIN)
    ax.set_ylim(Z_MAX - Z_MIN, 0)  # reverse y-axis
    ax.set_xticks([])
    ax.set_yticks([])

    # Cars from maps episode
    for v in sorted(episode["vehicles"], key=lambda d: d.get("isTarget", False)):
        ux, uz = v["position"]["x"], v["position"]["z"]
        gx, gy = unity_to_grid(ux, uz)
        prefab = v.get("name", "Black_Car_4")
        is_target = "Target" in v.get("name", "Black_Car_4")
        if "Black" in prefab:
            face = COLOR_MAP.get("black", "#888888")
        elif "Red" in prefab:
            face = COLOR_MAP.get("red", "#d62728")
        elif "White" in prefab:
            face = COLOR_MAP.get("white", "#dddddd")
        else:
            face = "#CCFF00"
        x0 = gx - CAR_W / 2.0
        y0 = gy - CAR_H / 2.0
        rect = patches.Rectangle(
            (x0, y0), CAR_W, CAR_H,
            linewidth=5.0 if is_target else 1.0,
            edgecolor="#CCFF00" if is_target else "#000000",
            facecolor=face,
            alpha=0.82 if is_target else 0.6
        )
        # print(f"Car at x0={x0:.2f}, y0={y0:.2f}, gx={gx:.2f}, gy={gy:.2f}, prefab={prefab}, target={is_target}")
        ax.add_patch(rect)

        # Visualize estimates if provided
    estimate_h = episode.get('estimate_h')
    estimate_x = episode.get('estimate_x')
    if estimate_h:
        gx_e, gy_e = unity_to_grid(estimate_h['x'], estimate_h['z'])
        ax.scatter(gx_e, gy_e, c='#FFD700', s=120, marker='*', label='Hyunsung Est', edgecolors='blue', linewidths=2, zorder=10)
    if estimate_x:
        gx_e, gy_e = unity_to_grid(estimate_x['x'], estimate_x['z'])
        ax.scatter(gx_e, gy_e, c='#FFA500', s=120, marker='P', label='Xuejing Est', edgecolors='orange', linewidths=2, zorder=10)

    # Agent icon & heading (0° = up)
    if draw_agent:
        # Allow overriding the agent state (position/rotation) for a given step
        agent_state = None
        if agent_override and isinstance(agent_override, dict) and \
           ("position" in agent_override) and ("rotation" in agent_override):
            agent_state = agent_override
        else:
            agent_state = episode.get("agent")

        if agent_state and ("position" in agent_state) and ("rotation" in agent_state):
            ax_u = agent_state["position"]["x"]
            az_u = agent_state["position"]["z"]
            gx_a, gy_a = unity_to_grid(ax_u, az_u)
            heading_deg = float(agent_state["rotation"].get("y", 0.0))
            ax.plot([gx_a], [gy_a], marker="o", markersize=2.5 if tiny else 4, color="#111111")
            # L = 2.0 if tiny else 2.5
            # th = math.radians(heading_deg)
            # dx =  L * np.sin(th)
            # dy = -L * np.cos(th)
            # ax.arrow(gx_a, gy_a, dx, dy, head_width=0.45 if tiny else 0.6,
            #          head_length=0.7 if tiny else 0.9, length_includes_head=True,
            #          color="#111111", linewidth=0.8)

    if label is not None:
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=6 if tiny else 7, color="#222",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.3))

def load_agent_data(file_path):
    if "/p" in file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else: 
        with open(file_path, "rb") as f:
            return pickle.load(f)


def parse_timestamp(timestamp_str):
    try:
        if '.' in timestamp_str:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        else:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None

def get_agent_trajectory(trial_dir):
    if 'agent' in trial_dir:
        agent_files = glob.glob(os.path.join(trial_dir, "agent_*.pkl"))
    else: 
        agent_files = glob.glob(os.path.join(trial_dir, "agent_*.json"))
    agent_files.sort()
    trajectory = []
    for file_path in agent_files:
        try:
            data = load_agent_data(file_path)
            if 'timestamp' in data:
                timestamp = parse_timestamp(data['timestamp'])
            # elif 'step' in data:
            #     timestamp = parse_timestamp(data['step'])
                if timestamp:
                    trajectory.append({
                        'timestamp': timestamp,
                        'position': data['position'],
                        'rotation': data['rotation'],
                        # 'audio_level': data.get('audioLevel', 0)
                    })
            else:
                trajectory.append({
                    'step': data['step'],
                    'position': data['position'],
                    'rotation': data['rotation'],
                    'prior': data['prior'],
                    'audio_like': data['audio_like'],
                    'visual_like': data['visual_like'],
                    'log_visual': data['log_visual'],
                    'log_audio': data['log_audio'],
                    'log_posterior': data['log_posterior'],
                })

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Sort trajectory by timestamp or step in ascending order
    if trajectory:
        if 'timestamp' in trajectory[0]:
            trajectory.sort(key=lambda x: x['timestamp'])
        elif 'step' in trajectory[0]:
            trajectory.sort(key=lambda x: x['step'])
    
    if len(trajectory) == 0:
        print(trial_dir, trajectory)
    return trajectory

def plot_trajectory(ax, trajectory, label, color,
                    show_rotation=False, show_fov=False, fov_deg=110.0,
                    fov_radius=None, fov_step=None, maps_style=True):
    if not trajectory:
        return

    # Convert to grid coords if maps_style, else use unity
    if maps_style:
        xs, zs = zip(*[unity_to_grid(p['position']['x'], p['position']['z']) for p in trajectory])
    else:
        xs = [p['position']['x'] for p in trajectory]
        zs = [p['position']['z'] for p in trajectory]

    line, = ax.plot(xs, zs, color=color, linewidth=2, alpha=0.5, label=label)
    ax.scatter(xs[0], zs[0], c='black', alpha=0.5, s=20, marker='o', label=f'{label} Start')
    ax.scatter(xs[-1], zs[-1], c=color, alpha=0.5, s=40, marker='^', label=f'{label} End')

    # FOV sectors (align heading convention)
    if show_fov:
        if fov_radius is None:
            # scale in current coord system
            fov_radius = 0.15 * np.hypot(max(xs)-min(xs), max(zs)-min(zs))
        step = max(1, len(trajectory) // 35) if fov_step is None else fov_step
        for i in range(0, len(trajectory), step):
            rot_y_deg = trajectory[i]['rotation']['y']
            # Matplotlib Wedge: 0° = +x axis CCW. Our heading 0° = up.
            # Convert: +y (up) equals 90° from +x. Also account for inverted y-limits later.
            center_deg = 90.0 - rot_y_deg if maps_style else rot_y_deg
            wedge = patches.Wedge((xs[i], zs[i]), fov_radius,
                                  center_deg - fov_deg/2, center_deg + fov_deg/2,
                                  alpha=0.15, color=color)
            ax.add_patch(wedge)

    # Straight heading ticks (no arrows), same style as visualize_maps
    if show_rotation:
        n_points = len(trajectory)
        step = max(1, n_points // 20)
        # step = 1
        line_len = 0.05 * np.hypot(max(xs)-min(xs), max(zs)-min(zs))
        for i in range(0, n_points, step):
            rot_y_deg = trajectory[i]['rotation']['y']
            th = np.deg2rad(rot_y_deg)
            # In maps-style grid: 0° up => dx=sin, dy=-cos (to match flipped y)
            if maps_style:
                u = np.sin(th)
                v = -np.cos(th)
            else:
                # Unity-style: keep previous forward convention (x right, z up)
                u = np.sin(th)
                v = np.cos(th)
            x0, z0 = xs[i], zs[i]
            x1, z1 = x0 + line_len*u, z0 + line_len*v
            ax.plot([x0, x1], [z0, z1], color=color, linewidth=1.5, alpha=0.8)

    # If maps_style, also match axes to visualize_maps
    if maps_style:
        ax.set_aspect('equal')
        ax.set_xlim(0, X_MAX - X_MIN)
        ax.set_ylim(Z_MAX - Z_MIN, 0)  # flip y like visualize_maps
        ax.set_xticks([])
        ax.set_yticks([])

    return line

def load_scene_data(trial_dir):
    scene_files = glob.glob(os.path.join(trial_dir, "scene_data_*.json"))
    if not scene_files:
        return None
    with open(scene_files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_vehicles_2d(ax, scene_data):
    if scene_data is None:
        return
    for vehicle in scene_data['vehicles']:
        pos = vehicle['position']
        name = vehicle['name']
        if 'SoundSource' in name:
            color = 'red'
            edge_color = 'orange'
        else:
            color = 'lightblue'
            edge_color = 'blue'
        car_width = 2.5
        car_length = 4.0
        x_left = pos['x'] - car_width/2
        z_bottom = pos['z'] - car_length/2
        rect = patches.Rectangle((x_left, z_bottom), car_width, car_length,
                               facecolor=color, edgecolor=edge_color, linewidth=5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(pos['x'], pos['z'], name, fontsize=8, ha='center', va='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black'))

def compute_metrics(trajectory, scene_data=None, estimate=None):
    if not trajectory:
        print("no trajectory: ", trajectory)
        return None
    start = trajectory[0]['position']
    end = trajectory[-1]['position']
    total_distance = np.sqrt((end['x'] - start['x'])**2 + (end['z'] - start['z'])**2)
    if 'timestamp' in trajectory[0]:
        start_time = trajectory[0]['timestamp']
        end_time = trajectory[-1]['timestamp']
        duration = (end_time - start_time).total_seconds()
    elif 'step' in trajectory[0]:
        # duration = trajectory[-1]['step'] / 9.0  # each step is around 0.11 s
        # duration = trajectory[-1]['step'] / 1.87  # each step is around 0.53 s
        duration = trajectory[-1]['step'] / 1.31  # each step is around 0.76 s

    avg_speed = total_distance / duration if duration > 0 else 0
    
    # Calculate accumulated head turns
    accumulated_head_turns = 0.0
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            prev_rotation = trajectory[i-1]['rotation']['y']
            curr_rotation = trajectory[i]['rotation']['y']
            # Calculate angular difference, accounting for wrap-around at 360°
            angle_diff = curr_rotation - prev_rotation
            # Normalize to [-180, 180] range
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            accumulated_head_turns += abs(angle_diff)
    
    # New metrics
    end_x, end_z = end['x'], end['z']
    target_vehicle = None
    end_to_target_dist = None
    estimate_correct = None
    estimate_to_target_dist = None
    
    # Find target vehicle
    if scene_data and 'vehicles' in scene_data:
        for vehicle in scene_data['vehicles']:
            if vehicle.get('isTarget') or 'Target' in vehicle.get('name', ''):
                target_vehicle = vehicle
                break
    
    if target_vehicle:
        target_pos = target_vehicle['position']
        end_to_target_dist = np.sqrt((end_x - target_pos['x'])**2 + (end_z - target_pos['z'])**2)
        
        # Check if estimate is correct (closest to target)
        if estimate and scene_data and 'vehicles' in scene_data:
            est_x, est_z = estimate['x'], estimate['z']
            estimate_to_target_dist = np.sqrt((est_x - target_pos['x'])**2 + (est_z - target_pos['z'])**2)
            
            # Find closest vehicle to estimate
            min_dist = float('inf')
            closest_vehicle = None
            for vehicle in scene_data['vehicles']:
                v_pos = vehicle['position']
                dist = np.sqrt((est_x - v_pos['x'])**2 + (est_z - v_pos['z'])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_vehicle = vehicle
            
            # Check if closest vehicle is the target
            if closest_vehicle:
                estimate_correct = (closest_vehicle.get('isTarget') or 'Target' in closest_vehicle.get('name', ''))
    
    return {
        'total_distance': total_distance,
        'duration': duration,
        'avg_speed': avg_speed,
        'accumulated_head_turns': accumulated_head_turns,
        'end_x': end_x,
        'end_z': end_z,
        'end_to_target_dist': end_to_target_dist,
        'estimate_correct': 1 if estimate_correct else 0,
        'estimate_to_target_dist': estimate_to_target_dist
    }

def dtw_distance(traj1, traj2):
    arr1 = np.array([[p['position']['x'], p['position']['z']] for p in traj1])
    arr2 = np.array([[p['position']['x'], p['position']['z']] for p in traj2])
    min_len = min(len(arr1), len(arr2))
    arr1_rs = resample(arr1, min_len)
    arr2_rs = resample(arr2, min_len)
    dist_matrix = cdist(arr1_rs, arr2_rs, metric='euclidean')
    n, m = dist_matrix.shape
    dtw = np.zeros((n+1, m+1)) + np.inf
    dtw[0,0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist_matrix[i-1, j-1]
            dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    return dtw[n,m]


def _imshow_panel(ax, arr, title):
    if arr is None:
        ax.set_axis_off()
        return
    im = ax.imshow(arr, origin='lower', aspect='auto')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# --- Density / occupancy helpers ---

def _collect_xy_grid(traj):
    """Return Nx2 array of (gx, gy) for a single trajectory using unity_to_grid."""
    if not traj:
        return None
    xs, ys = [], []
    for p in traj:
        try:
            gx, gy = unity_to_grid(p['position']['x'], p['position']['z'])
            xs.append(gx); ys.append(gy)
        except Exception:
            continue
    if not xs:
        return None
    return np.column_stack([xs, ys])


def _density_difference(human_xy, agent_xy, bins_w=GRID_W, bins_h=GRID_H):
    """Compute 2D densities for human and agent positions on the grid and their difference.
    Returns (H, A, D, xedges, yedges). If either input is None or empty, returns None.
    """
    if human_xy is None and agent_xy is None:
        return None
    # Bin edges aligned to grid coordinates [0..GRID_*]
    xedges = np.linspace(0, GRID_W, bins_w + 1)
    yedges = np.linspace(0, GRID_H, bins_h + 1)

    def _hist(xy):
        if xy is None or len(xy) == 0:
            return np.zeros((bins_w, bins_h)).T  # (y,x) later transposed in imshow
        H, _, _ = np.histogram2d(xy[:,0], xy[:,1], bins=[xedges, yedges], density=True)
        return H

    H = _hist(human_xy)
    A = _hist(agent_xy)
    D = H - A
    return H, A, D, xedges, yedges


def _plot_density_triptych(H, A, D, xedges, yedges, title_prefix, outfile):
    """Plot Human density, Agent density, and Human-Agent difference and save to outfile."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im0 = axs[0].imshow(H.T, origin='lower', aspect='equal', extent=extent)
    axs[0].set_title(f'{title_prefix} Human density')
    cb0 = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(A.T, origin='lower', aspect='equal', extent=extent)
    axs[1].set_title(f'{title_prefix} Agent density')
    cb1 = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    vmax = np.max(np.abs(D)) if np.max(np.abs(D)) > 0 else 1.0
    im2 = axs[2].imshow(D.T, origin='lower', aspect='equal', cmap='coolwarm',
                        vmin=-vmax, vmax=vmax, extent=extent)
    axs[2].set_title(f'{title_prefix} Human − Agent')
    cb2 = plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)

def save_belief_grid_for_step(step_dict, out_path):
    """
    Save a 2x4 grid of belief/likelihood heatmaps from a single trajectory step dict.
    Expects keys: 'prior', 'audio_like', 'visual_like', 'log_visual', 'log_audio', 'log_posterior'
    Also plots an estimate point if 'est_world_position' (or 'est_world position') exists.
    """
    if step_dict is None or not isinstance(step_dict, dict):
        return

    # Pull arrays if present
    prior         = step_dict.get("prior")
    audio_like    = step_dict.get("audio_like")
    visual_like   = step_dict.get("visual_like")
    log_visual    = step_dict.get("log_visual")
    log_audio     = step_dict.get("log_audio")
    log_posterior = step_dict.get("log_posterior")

    # If none of the arrays exist, skip
    has_any = any(x is not None for x in [prior, audio_like, visual_like, log_visual, log_audio, log_posterior])
    if not has_any:
        return

    # Create output dir
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    axes = axes.ravel()

    _imshow_panel(axes[0], prior,         "prior")
    _imshow_panel(axes[1], audio_like,    "audio_like")
    _imshow_panel(axes[2], visual_like,   "visual_like")
    _imshow_panel(axes[3], log_visual,    "log_visual")
    _imshow_panel(axes[4], log_audio,     "log_audio")
    _imshow_panel(axes[5], log_posterior, "log_posterior")

    # 7th panel: show estimate position if available
    est = step_dict.get("est_world_position", step_dict.get("est_world position"))
    ax_est = axes[6]
    if est is not None and isinstance(est, dict) and {"x","z"}.issubset(est.keys()):
        # If we know the array shapes, set a reasonable canvas using one of the arrays
        # Otherwise just plot a simple point with generic limits.
        ref = None
        for candidate in [prior, audio_like, visual_like, log_visual, log_audio, log_posterior]:
            if isinstance(candidate, np.ndarray):
                ref = candidate
                break
        ax_est.set_title("est_world_position", fontsize=9)
        ax_est.set_xticks([]); ax_est.set_yticks([])
        if isinstance(ref, np.ndarray) and ref.ndim == 2:
            h, w = ref.shape
            ax_est.set_xlim(0, w); ax_est.set_ylim(0, h)
        else:
            ax_est.set_xlim(0, 1); ax_est.set_ylim(0, 1)
        # Plot a marker; we don't know mapping from world (x,z) → grid here, so just annotate values.
        ax_est.scatter([0.5], [0.5], marker='*', s=120)
        ax_est.text(0.02, 0.95, f"x={est['x']:.2f}\nz={est['z']:.2f}",
                    transform=ax_est.transAxes, va='top', ha='left',
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    else:
        ax_est.set_axis_off()

    # 8th panel left blank (or you can repurpose if you later add another map)
    axes[7].set_axis_off()

    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def process_single(pid: str, trial: str, show_rotation: bool=False, show_fov: bool=False, fov_deg: float=110.0, fov_radius=None, fov_step=None):
    """
    Process and plot a single participant trial located at ./data/{pid}/{trial}.
    Saves:
      - plots/single_{pid}_{trial}.png : scene + trajectory
      - plots/beliefs_{pid}_{trial}.png: 2x4 belief/likelihood grid (if arrays exist)
    """
    base_dir = os.path.join('./data', pid, str(trial))
    if not os.path.isdir(base_dir):
        print(f"[single] Directory not found: {base_dir}")
        return

    # Load scene
    scene_data = load_scene_data(base_dir)
    if scene_data is None:
        print(f"[single] No scene_data_*.json found under: {base_dir}")
        return

    # Load trajectory
    traj = get_agent_trajectory(base_dir)
    if not traj:
        print(f"[single] No agent_*.pkl/json frames found under: {base_dir}")
        return

    # If this episode has a separate estimate file, attach it for plotting label parity
    est_files = glob.glob(os.path.join(base_dir, 'estimate*.json'))
    if est_files:
        try:
            with open(est_files[0], 'r', encoding='utf-8') as f:
                est_json = json.load(f)
            # Agents store {'est_world_position': {...}}
            # Humans might store {'x': ..., 'z': ...}
            if 'agent' in pid:
                scene_data[f'estimate_{pid}'] = est_json
            else:
                scene_data[f'estimate_{pid}'] = est_json
        except Exception as _e:
            pass

    # Plot scene + trajectory
    os.makedirs('plots', exist_ok=True)
    fig = plt.figure(figsize=(7, 8))
    ax = plt.subplot(111)
    draw_episode_like_maps(ax, scene_data, show_grid=False, tiny=False)

    # Choose a stable color for the pid if available
    colors = {
        'p01': 'orange', 'p02': 'blue', 'p03': 'green', 'p04': 'purple',
        'p05': 'pink', 'p06': 'black', 'p07': 'lightblue', 'p08': 'lightgreen',
        'p09': 'tab:olive', 'p10': 'tab:cyan', 'p11': 'tab:gray', 'p12': 'tab:red',
        'agent_1': 'brown', 'agent_2': 'magenta', 'agent_3': 'gray', 'agent_4': 'cyan',
        'agent_5': 'olive', 'agent_6': 'pink', 'agent_7': 'lightcoral', 'agent_8': 'khaki',
        'agent_9': 'tab:purple', 'agent_10': 'tab:green', 'agent_11': 'tab:blue', 'agent_12': 'tab:orange'
    }
    color = colors.get(pid, None) or None
    plot_trajectory(ax, traj, pid, color,
                    show_rotation=show_rotation,
                    show_fov=show_fov, fov_deg=fov_deg,
                    fov_radius=fov_radius, fov_step=fov_step)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title(f'{pid} Trial {trial}: Trajectory and Vehicles')
    # fig.legend(loc='outside upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = f'plots/single_{pid}_{trial}.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[single] Saved {out_path}")

    # Save belief/likelihood grid from the last step (if arrays exist)
    try:
        last_step = traj[-1] if isinstance(traj[-1], dict) else None
        if last_step:
            save_belief_grid_for_step(last_step, f'plots/beliefs_{pid}_{trial}.png')
            print(f"[single] Saved plots/beliefs_{pid}_{trial}.png")
    except Exception as e:
        print(f"[single] Failed to save belief grid for {pid} trial {trial}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Compare trajectories of Hyunsung and Xuejing.')
    parser.add_argument('--show-rotation', action='store_true', default=False,
                        help='Show rotation vectors on trajectory plots')
    parser.add_argument('--show-fov', action='store_true', default=False,
                        help='Show 110° Field-of-View (FOV) sectors on trajectory plots')
    parser.add_argument('--fov-deg', type=float, default=110.0,
                        help='FOV angle in degrees (default 110.0)')
    parser.add_argument('--fov-radius', type=float, default=None,
                        help='FOV radius in meters (default auto-computed)')
    parser.add_argument('--fov-step', type=int, default=None,
                        help='Sampling step for FOV wedges (default auto-computed)')
    parser.add_argument('--pid', type=str, default=None,
                        help='Single participant ID to process (e.g., p01, agent_3)')
    parser.add_argument('--trial', type=str, default=None,
                        help='Trial folder name under ./data/{pid}/ (e.g., 5). For humans this is the episode index; for agents it is the numeric trial dir.')
    parser.add_argument('--model-name', type=str, required=True, help="model name to find test logs")
    args = parser.parse_args()

    # --- Exclude trials per participant ---
    exclude_trials = {
        'p01': [9, 25, 131, 161],
        'p02': [8, 31, 49, 53, 74, 105, 109, 143, 255],
        'p03': [],
        'p04': [0],
        'p05': [],
        'p06': [97],
        'p07': [],
        'p08': [],
        'p09': [],
        'p10': [20, 28, 49, 51, 55, 57, 60, 64, 68, 100, 104, 106, 117, 121, 140, 154, 164, 220, 232, 238, 241, 252, 255, 258],
        'p11': [72, 119],
        'p12': [0, 15, 25, 41, 49, 58, 73, 125, 169, 200],
    }

    # --- Single participant/trial mode ---
    if args.pid is not None and args.trial is not None:
        process_single(args.pid, args.trial,
                       show_rotation=args.show_rotation,
                       show_fov=args.show_fov,
                       fov_deg=args.fov_deg,
                       fov_radius=args.fov_radius,
                       fov_step=args.fov_step)
        return

    # Humans use maps.json indexing; agents are matched by scene_data['map_id']
    humans = ['p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12']
    agents = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6', 'agent_7', 'agent_8', 'agent_9', 'agent_10', 'agent_11', 'agent_12']
    participants = humans + agents
    participant_dirs = {}
    for pid in humans:
        participant_dirs[pid] = f'analysis/data/{pid}'

    for pid in agents:
        # participant_dirs[pid] = f'./data/agent-turn90-step1/{pid}'
        participant_dirs[pid] = f'test_logs/{args.model_name}/{pid}'

    results = []
    # Load maps.json
    with open('maps/maps.json', 'r', encoding='utf-8') as f:
        maps_data = json.load(f)
        # Build a global lookup so we can attach conditions for agents as well
        map_id_to_conditions = {}
        for _pid, _maps in maps_data.items():
            if not isinstance(_maps, list):
                continue
            for _m in _maps:
                if not isinstance(_m, dict):
                    continue
                # Accept either 'map_id' or 'id'
                cid = _m.get('map_id', _m.get('id'))
                if cid is None:
                    continue
                if cid not in map_id_to_conditions:
                    # Accept either 'conditions' or 'condition'
                    map_id_to_conditions[cid] = _m.get('conditions', _m.get('condition', {}))

    # Collect all map_ids across all participants
    all_map_ids = set()
    participant_maps = {}
    participant_mapid_to_idx = {}  # for humans (p01, p02, ...)
    agent_mapid_to_dir = {}        # for agents (agent_1, agent_2, ...)

    for pid in participants:
        if pid.startswith('p'):  # HUMAN: uses maps.json
            maps = maps_data.get(pid, [])
            participant_maps[pid] = maps
            mapid_to_idx = {m['map_id']: i for i, m in enumerate(maps) if 'map_id' in m}
            participant_mapid_to_idx[pid] = mapid_to_idx
            all_map_ids.update(mapid_to_idx.keys())
        else:
            # AGENT: scan its data directory and read scene_data_*.json to get map_id
            base_dir = participant_dirs[pid]
            mapping = {}
            if os.path.isdir(base_dir):
                # trial directories are numeric (e.g., ./data/agent_1/1, ./data/agent_1/2, ...)
                for trial_dir in sorted(glob.glob(os.path.join(base_dir, '*'))):
                    if not os.path.isdir(trial_dir):
                        continue
                    scene_files = glob.glob(os.path.join(trial_dir, 'scene_data_*.json'))
                    if not scene_files:
                        continue
                    try:
                        with open(scene_files[0], 'r', encoding='utf-8') as sf:
                            scene_json = json.load(sf)
                        mid = scene_json.get('map_id', None)
                        if mid is None and isinstance(scene_json.get('agent'), dict):
                            mid = scene_json['agent'].get('map_id')
                        if mid is not None and mid not in mapping:
                            mapping[mid] = trial_dir
                            all_map_ids.add(mid)
                    except Exception as e:
                        print(f"Error reading scene file for {pid} in {trial_dir}: {e}")
            agent_mapid_to_dir[pid] = mapping

    all_map_ids = sorted(all_map_ids)

    # Collect per-map plotting payloads so we can group by conditions later
    grouped_maps = {}

    for map_id in all_map_ids:
        # Gather episode index for each participant for this map_id
        idx_by_pid = {}
        dirs_by_pid = {}
        conditions_by_pid = {}

        for pid in participants:
            if pid.startswith('p'):
                idx = participant_mapid_to_idx.get(pid, {}).get(map_id)
                idx_by_pid[pid] = idx
                dirs_by_pid[pid] = os.path.join(participant_dirs[pid], str(idx)) if idx is not None else None
                conditions_by_pid[pid] = (participant_maps[pid][idx]['conditions']
                                          if idx is not None and idx < len(participant_maps[pid]) else {})
            else:
                # Agents: match directory by scene_data['map_id']
                trial_dir = agent_mapid_to_dir.get(pid, {}).get(map_id)
                dirs_by_pid[pid] = trial_dir
                idx_by_pid[pid] = os.path.basename(trial_dir) if trial_dir else None
                conditions_by_pid[pid] = map_id_to_conditions.get(map_id, {})
                if not conditions_by_pid[pid]:
                    # Try to load conditions from the agent's scene file if present
                    try:
                        if trial_dir and os.path.isdir(trial_dir):
                            _sf = glob.glob(os.path.join(trial_dir, 'scene_data_*.json'))
                            if _sf:
                                with open(_sf[0], 'r', encoding='utf-8') as _f:
                                    _sj = json.load(_f)
                                # Accept 'conditions' or 'condition'
                                conditions_by_pid[pid] = _sj.get('conditions', _sj.get('condition', {})) or {}
                    except Exception as _e:
                        pass
        # Only process if at least two participants have valid dirs and dirs exist
        valid_participants = []
        for pid in participants:
            if dirs_by_pid[pid] and os.path.isdir(dirs_by_pid[pid]):
                trial_id = idx_by_pid[pid]
                # Skip if trial is in exclusion list
                if pid in exclude_trials and trial_id is not None:
                    try:
                        trial_int = int(trial_id)
                    except Exception:
                        trial_int = None
                    if trial_int is not None and trial_int in exclude_trials[pid]:
                        print(f"Exclude trial {trial_id} for participant {pid}")
                        continue
                valid_participants.append(pid)
        # if len(valid_participants) < 2:
        #     print("valid participant")
        #     continue
        # Load trajectories, scene, and estimates for all valid participants
        trajs = {pid: get_agent_trajectory(dirs_by_pid[pid]) for pid in valid_participants}
        # Only plot if at least two have non-empty trajectories
        # if sum(1 for t in trajs.values() if t) < 2:
        #     print("  skipping: less than two with trajectories")
        #     # print(trajs.values())
        #     continue
        scenes = {pid: load_scene_data(dirs_by_pid[pid]) for pid in valid_participants}
        def load_estimate(ep_dir):
            est_files = glob.glob(os.path.join(ep_dir, 'estimate*.json'))
            if est_files:
                with open(est_files[0], 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        # Use the first available scene as the base
        scene_data = None
        for pid in valid_participants:
            if scenes[pid] is not None:
                scene_data = scenes[pid]
                break
        if scene_data is None:
            print("no scene data")
            continue
        # Add each participant's estimate to scene_data for plotting
        for pid in valid_participants:
            scene_data[f'estimate_{pid}'] = load_estimate(dirs_by_pid[pid])
        
        metric_by_pid = {}
        for pid in valid_participants:
            est = scene_data.get(f'estimate_{pid}')
            if est is not None and 'agent' in pid and isinstance(est, dict):
                est = est.get('est_world_position', est)
            metric_by_pid[pid] = compute_metrics(trajs[pid], scene_data, est)

        # Save belief/likelihood grids for each participant (if arrays exist on last step)
        # for pid in valid_participants:
        #     traj = trajs.get(pid)
        #     if traj and isinstance(traj[-1], dict):
        #         out_path = f"plots/beliefs_map_{map_id}_{pid}.png"
        #         try:
        #             save_belief_grid_for_step(traj[-1], out_path)
        #         except Exception as e:
        #             print(f"Failed to save belief grid for {pid} map {map_id}: {e}")
        # Save results
        for pid in valid_participants:
            m = metric_by_pid[pid]
            # Use dtw to p01 and p02 if available, else first other participant
            # dtw_val = None
            # for ref_pid in ['p01', 'p02']:
            #     if (pid, ref_pid) in dtw_pairs:
            #         dtw_val = dtw_pairs[(pid, ref_pid)]
            #         break
            #     if (ref_pid, pid) in dtw_pairs:
            #         dtw_val = dtw_pairs[(ref_pid, pid)]
            #         break
            # if dtw_val is None and dtw_pairs:
            #     dtw_val = next(iter(dtw_pairs.values()))
            results.append({
                'map_id': map_id,
                'participant': pid,
                'agent_type': 'agent' if 'agent' in pid else 'human',
                'distance': m['total_distance'],
                'duration': m['duration'],
                'speed': m['avg_speed'],
                # 'dtw': dtw_val,
                'angle': conditions_by_pid[pid].get('angle', ''),
                'num_cars': conditions_by_pid[pid].get('num_cars', ''),
                'distractors': conditions_by_pid[pid].get('distractors', ''),
                # 'noise': conditions_by_pid[pid].get('noise', ''),
                'accumulated_head_turns': m['accumulated_head_turns'],
                'end_x': m['end_x'],
                'end_z': m['end_z'],
                'end_to_target_dist': m['end_to_target_dist'],
                'estimate_correct': m['estimate_correct'],
                'estimate_to_target_dist': m['estimate_to_target_dist'],
                'idx': idx_by_pid[pid]
            })
        # Stash for grouped plotting by conditions
        cond = map_id_to_conditions.get(map_id, {})
        cond_key = (
            cond.get('angle', ''),
            cond.get('num_cars', ''),
            cond.get('distractors', ''),
        )
        colors = {
            'p01': 'orange', 'p02': 'blue', 'p03': 'green', 'p04': 'purple',
            'p05': 'pink', 'p06': 'black', 'p07': 'brown', 'p08': 'magenta',
            'p09': 'gray', 'p10': 'cyan', 'p11': 'olive', 'p12': 'pink',
            'agent_1': 'orange', 'agent_2': 'blue', 'agent_3': 'green', 'agent_4': 'purple',
            'agent_5': 'pink', 'agent_6': 'black', 'agent_7': 'brown', 'agent_8': 'magenta',
            'agent_9': 'gray', 'agent_10': 'cyan', 'agent_11': 'olive', 'agent_12': 'pink'
        }
        human_participants = [p for p in valid_participants if p.startswith('p')]
        agent_participants = [p for p in valid_participants if p.startswith('agent')]
        # Compute per-map accuracies for humans and agents
        def _safe_mean(vals):
            vals = [v for v in vals if v is not None]
            return (sum(vals) / len(vals)) if vals else None
        human_acc = _safe_mean([metric_by_pid.get(p, {}).get('estimate_correct') for p in human_participants])
        agent_acc = _safe_mean([metric_by_pid.get(p, {}).get('estimate_correct') for p in agent_participants])
        correct_by_pid = {p: (metric_by_pid.get(p, {}) or {}).get('estimate_correct') for p in valid_participants}
        payload = {
            'map_id': map_id,
            'scene_data': scene_data,
            'trajs': {pid: trajs[pid] for pid in valid_participants},
            'human_participants': human_participants,
            'agent_participants': agent_participants,
            'colors': colors,
            'human_acc': human_acc,
            'agent_acc': agent_acc,
            'correct_by_pid': correct_by_pid,
            'trial_idx_by_pid': {pid: idx_by_pid.get(pid) for pid in valid_participants},
        }
        grouped_maps.setdefault(cond_key, []).append(payload)


    # After collecting all maps, render grouped figures by condition
    for cond_key, maps_list in grouped_maps.items():
        angle, num_cars, distractors = cond_key

        def _render(maps_subset, suffix):
            # Chunk into pages of up to 10 maps each
            for page_idx in range(0, len(maps_subset), 10):
                chunk = maps_subset[page_idx:page_idx+10]
                # Arrange as 4x5 grid of axes: two maps per row, each map uses 2 columns (H, A)
                n_maps = len(chunk)
                if n_maps == 0:
                    continue
                n_rows = (n_maps + 1) // 2  # two maps per row
                # For 'all', we show 4 panels per map (HS, AS, HF, AF) => 8 columns total
                if suffix == 'all':
                    n_cols = 8
                else:
                    n_cols = 4  # each map gets two columns: [Human, Agent]
                fig_w = 26 if n_cols == 8 else 18
                fig_h = max(3.8 * n_rows, 7)
                fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_w, fig_h), squeeze=False)
                fig.suptitle(f'Condition: angle={angle}, cars={num_cars}, distractors={distractors}  [{suffix}]', fontsize=14)
                page_legend_ids = set()

                # Plot each map: compute row/col pair
                for i, item in enumerate(chunk):
                    r = i // 2
                    scene_data = item['scene_data']
                    corr = item.get('correct_by_pid', {})
                    colors = item['colors']

                    # Helper to star estimates
                    def _star_est(ax, pid, color=None, marker='*'):
                        est = scene_data.get(f'estimate_{pid}')
                        est_pos = None
                        if isinstance(est, dict):
                            if isinstance(est.get('est_world_position'), dict):
                                d = est['est_world_position']
                                est_pos = {'x': d.get('x'), 'z': d.get('z')}
                            elif 'x' in est and 'z' in est:
                                est_pos = {'x': est.get('x'), 'z': est.get('z')}
                        if est_pos:
                            ex, ez = unity_to_grid(est_pos['x'], est_pos['z'])
                            ax.scatter([ex], [ez], marker=marker, alpha=0.5, s=40, c=color if color is not None else colors.get(pid, '#444'),
                                       edgecolors='black', linewidths=0.8, zorder=12)

                    if suffix == 'all':
                        # Four subpanels per map
                        base = (i % 2) * 4
                        ax_hs, ax_as = axs[r, base + 0], axs[r, base + 1]
                        ax_hf, ax_af = axs[r, base + 2], axs[r, base + 3]

                        for ax in (ax_hs, ax_as, ax_hf, ax_af):
                            draw_episode_like_maps(ax, scene_data, show_grid=False, tiny=False)

                        h_succ = [p for p in item['human_participants'] if corr.get(p) == 1]
                        a_succ = [p for p in item['agent_participants'] if corr.get(p) == 1]
                        h_fail = [p for p in item['human_participants'] if corr.get(p) == 0]
                        a_fail = [p for p in item['agent_participants'] if corr.get(p) == 0]

                        
                        # for pid in item['human_participants']:
                        #     plot_trajectory(ax_hf, item['trajs'][pid], pid, colors.get(pid), False, False)
                        #     _star_est(ax_hf, pid); page_legend_ids.add(pid)
                        # for pid in item['agent_participants']:
                        #     plot_trajectory(ax_af, item['trajs'][pid], pid, colors.get(pid), False, False)
                        #     _star_est(ax_af, pid); page_legend_ids.add(pid)
                        
                        for pid in h_fail:
                            plot_trajectory(ax_hf, item['trajs'][pid], pid, 'red', False, False)
                            _star_est(ax_hf, pid, 'red', marker='X'); page_legend_ids.add(pid)
                        for pid in a_fail:
                            plot_trajectory(ax_af, item['trajs'][pid], pid, 'red', False, False)
                            _star_est(ax_af, pid, 'red', marker='X'); page_legend_ids.add(pid)

                        for pid in h_succ:
                            plot_trajectory(ax_hf, item['trajs'][pid], pid, 'green', False, False)
                            _star_est(ax_hf, pid, 'green', marker='o'); page_legend_ids.add(pid)
                        for pid in a_succ:
                            plot_trajectory(ax_af, item['trajs'][pid], pid, 'green', False, False)
                            _star_est(ax_af, pid, 'green', marker='o'); page_legend_ids.add(pid)

                        
                        ax_hs.set_title(f"Map {item['map_id']}: H Success", fontsize=10)
                        ax_as.set_title(f"Map {item['map_id']}: A Success", fontsize=10)
                        ax_hf.set_title(f"Map {item['map_id']}: H All", fontsize=10)
                        ax_af.set_title(f"Map {item['map_id']}: A All", fontsize=10)
                        for ax in (ax_hs, ax_as, ax_hf, ax_af):
                            ax.grid(True, alpha=0.2)
                        # Show overall per-map accuracy percentages
                        h_acc = item.get('human_acc')
                        a_acc = item.get('agent_acc')
                        def _fmt(v):
                            return (f"{100.0 * v:.0f}%" if isinstance(v, (int, float)) else "NA")
                        txt = f"H acc: {_fmt(h_acc)}  |  A acc: {_fmt(a_acc)}"
                        ax_hs.text(0.02, 0.02, txt, transform=ax_hs.transAxes, fontsize=9,
                                   ha='left', va='bottom', color='#222',
                                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.2))
                        ax_as.text(0.02, 0.02, txt, transform=ax_as.transAxes, fontsize=9,
                                   ha='left', va='bottom', color='#222',
                                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.2))
                    else:
                        # Two subpanels per map as before
                        c0 = (i % 2) * 2
                        ax_h, ax_a = axs[r, c0], axs[r, c0 + 1]
                        draw_episode_like_maps(ax_h, scene_data, show_grid=False, tiny=False)
                        draw_episode_like_maps(ax_a, scene_data, show_grid=False, tiny=False)

                        # Select participants based on failure pages
                        h_sel = item['human_participants']
                        a_sel = item['agent_participants']
                        if suffix == 'human-failure':
                            h_sel = [p for p in h_sel if corr.get(p) == 0]
                        if suffix == 'agent-failure':
                            a_sel = [p for p in a_sel if corr.get(p) == 0]

                        for pid in h_sel:
                            plot_trajectory(ax_h, item['trajs'][pid], pid, colors.get(pid), False, False)
                            page_legend_ids.add(pid)
                            if suffix == 'human-failure':
                                _star_est(ax_h, pid)
                        for pid in a_sel:
                            plot_trajectory(ax_a, item['trajs'][pid], pid, colors.get(pid), False, False)
                            page_legend_ids.add(pid)
                            if suffix == 'agent-failure':
                                _star_est(ax_a, pid)
                    # Titles
                    if suffix != 'all':
                        ax_h.set_title(f'Map {item["map_id"]}: Humans', fontsize=11)
                        ax_a.set_title(f'Map {item["map_id"]}: Agents', fontsize=11)
                        ax_h.grid(True, alpha=0.2)
                        ax_a.grid(True, alpha=0.2)
                        # Accuracy text overlay (omit in all view to reduce clutter)
                        h_acc = item.get('human_acc')
                        a_acc = item.get('agent_acc')
                        def _fmt(v):
                            return (f"{100.0 * v:.0f}%" if isinstance(v, (int, float)) else "NA")
                        txt = f"H acc: {_fmt(h_acc)}  |  A acc: {_fmt(a_acc)}"
                        ax_h.text(0.02, 0.02, txt, transform=ax_h.transAxes, fontsize=10,
                                  ha='left', va='bottom', color='#222',
                                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
                    # If rendering failure pages, list failed agents with compact trial IDs
                    if suffix == 'agent-failure':
                        idx_map = item.get('trial_idx_by_pid', {})
                        failed_agents = a_sel
                        if failed_agents:
                            lines = [f"{p}:t{idx_map.get(p)}" for p in failed_agents]
                            msg = "Failed:\n" + "\n".join(lines)
                            # Place to the left outside the agent axes
                            ax_a.text(-0.14, 1.0, msg, transform=ax_a.transAxes, fontsize=9,
                                      ha='left', va='top', color='#8b0000', clip_on=False,
                                      bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.2))
                            # Also mark which vehicle each failed agent likely guessed (closest to estimate)
                            def _extract_est(scene_data, pid):
                                est = scene_data.get(f'estimate_{pid}')
                                if isinstance(est, dict):
                                    if isinstance(est.get('est_world_position'), dict):
                                        d = est['est_world_position']
                                        return {'x': d.get('x'), 'z': d.get('z')}
                                    if 'x' in est and 'z' in est:
                                        return {'x': est.get('x'), 'z': est.get('z')}
                                return None
                            def _closest_vehicle(scene_data, pos):
                                if not pos or 'x' not in pos or 'z' not in pos:
                                    return None
                                best, best_d = None, None
                                for v in scene_data.get('vehicles', []):
                                    vx, vz = v['position']['x'], v['position']['z']
                                    d = (vx - pos['x'])**2 + (vz - pos['z'])**2
                                    if best is None or d < best_d:
                                        best, best_d = v, d
                                return best
                            for fp in failed_agents:
                                est_pos = _extract_est(scene_data, fp)
                                v = _closest_vehicle(scene_data, est_pos)
                                if v is None:
                                    continue
                                gx, gy = unity_to_grid(v['position']['x'], v['position']['z'])
                                x0 = gx - CAR_W / 2.0
                                y0 = gy - CAR_H / 2.0
                                hl = patches.Rectangle((x0, y0), CAR_W, CAR_H,
                                                       linewidth=2.8, edgecolor='#d62728',
                                                       facecolor='none', linestyle='--')
                                ax_a.add_patch(hl)
                                # Mark exact estimate location with a star
                                if est_pos and 'x' in est_pos and 'z' in est_pos:
                                    ex, ez = unity_to_grid(est_pos['x'], est_pos['z'])
                                    ax_a.scatter([ex], [ez], marker='*', s=90, c='#d62728',
                                                 edgecolors='black', linewidths=0.8, zorder=12)

                    # Human failure page: list failed humans above the human panel
                    if suffix == 'human-failure':
                        idx_map = item.get('trial_idx_by_pid', {})
                        failed_humans = h_sel
                        if failed_humans:
                            lines = [f"{p}:t{idx_map.get(p)}" for p in failed_humans]
                            msg = "Failed:\n" + "\n".join(lines)
                            ax_h.text(-0.14, 1.0, msg, transform=ax_h.transAxes, fontsize=9,
                                      ha='left', va='top', color='#8b0000', clip_on=False,
                                      bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.2))

                # Global legend for colors used in this page
                if page_legend_ids:
                    handles = [Line2D([0], [0], color=chunk[0]['colors'].get(pid, '#444'), lw=3) for pid in sorted(page_legend_ids)]
                    labels = sorted(page_legend_ids)
                    # labels = [l.replace("agent", "a") for l in labels]
                    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5), frameon=True, title='Participants')
                    # fig.subplots_adjust(right=0.85)

                # Turn off any unused map slots in the last row
                total_slots = n_rows * 2  # two maps per row
                for i in range(n_maps, total_slots):
                    r0 = i // 2
                    if suffix == 'all':
                        cbase = (i % 2) * 4
                        for dc in range(4):
                            axs[r0, cbase + dc].set_axis_off()
                    else:
                        c0 = (i % 2) * 2
                        axs[r0, c0].set_axis_off()
                        axs[r0, c0 + 1].set_axis_off()

                plt.subplots_adjust(left=0.08, wspace=0.25, hspace=0.4)
                plt.tight_layout(rect=[0.06, 0.04, 0.9, 0.96])
                out_name = f'analysis/plots/compare_condition_angle{angle}_cars{num_cars}_dist{distractors}_{suffix}_p{page_idx//10 + 1}.png'
                plt.savefig(out_name, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"[grouped] Saved {out_name}")

        # Render all, and split by success/failure for humans and agents
        _render(maps_list, 'all')
        # success_thres = 0.5
        # human_success = [m for m in maps_list if m.get('human_acc') is not None and m['human_acc'] >= success_thres]
        # human_failure = [m for m in maps_list if m.get('human_acc') is not None and m['human_acc'] < success_thres]
        # agent_success = [m for m in maps_list if m.get('agent_acc') is not None and m['agent_acc'] >= success_thres]
        # agent_failure = [m for m in maps_list if m.get('agent_acc') is not None and m['agent_acc'] < success_thres]
        # _render(human_success, 'human-success')
        # _render(human_failure, 'human-failure')
        # _render(agent_success, 'agent-success')
        # _render(agent_failure, 'agent-failure')

    # Save results
    df = pd.DataFrame(results)
    # Add idx column for each participant (for compatibility)
    cols = ['map_id','participant','agent_type','distance','duration','speed','angle','num_cars','distractors','accumulated_head_turns','end_x','end_z','end_to_target_dist','estimate_correct','estimate_to_target_dist','idx']
    df = df[cols]
    df.to_csv('compare_human_participants_metrics_by_map.csv', index=False)
    print('Saved metrics to compare_human_participants_metrics_by_map.csv')

if __name__ == "__main__":
    main()
