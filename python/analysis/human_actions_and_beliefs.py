#!/usr/bin/env python3
import os
import sys
import glob
import json
import math
import re
import argparse
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root on path for utils imports
_CUR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.constants import THETA_GRID, R_GRID
from utils.audio_features import compute_itd_ild_from_wav
from scipy.ndimage import gaussian_filter1d
from analysis.plot_beliefs import save_belief_grid_for_step
from analysis.compare_traj import load_scene_data, unity_to_grid, draw_episode_like_maps, parse_timestamp, get_agent_trajectory
import json as _json
import pickle as _pickle

# Set Roboto as the default sans-serif font
# plt.rcParams['font.family'] = 'sans-serif' 
# plt.rcParams['font.sans-serif'] = ['Helvetica'] 


ACTION_ORDER = ['turn_left', 'turn_right', 'step_forward', 'stay', 'commit']
ACTION_TO_Y = {a: i for i, a in enumerate(ACTION_ORDER)}


def load_frames(ep_dir: str) -> List[Dict]:
    # def extract_num(path):
    #     # Extract the number after "agent_"
    #     m = re.search(r'agent_data_(\d+)', os.path.basename(path))
    #     return int(m.group(1)) if m else float('inf')
    files = sorted(glob.glob(os.path.join(ep_dir, 'agent_data_*.json')))
    frames = []

    scene = load_scene_data(ep_dir)
    init_beliefs = np.log(np.ones((R_GRID.shape[0], THETA_GRID.shape[0])) / (R_GRID.shape[0] * THETA_GRID.shape[0]) + 1e-12)
    init_frame = {
            "timestamp": "init",
            "position": scene['agent']['position'],
            "rotation": scene['agent']['rotation'],
            "log_visual": init_beliefs,
            "log_audio": init_beliefs,
            "log_posterior": init_beliefs,
            "log_belief": init_beliefs,
        }
    init_frame["visualData"] = init_frame.copy()
    init_frame["visualData"]["detectedObjects"] = []
    frames.append(
        init_frame
    )
    for fp in files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                frames.append(json.load(f))
        except Exception as e:
            print(f"[warn] failed to read {fp}: {e}")
    return frames

def load_agent_frames_generic(ep_dir: str) -> List[Dict]:
    def extract_num(path):
        # Extract the number after "agent_"
        m = re.search(r'agent_(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else float('inf')
        
    """Load agent trial frames supporting both agent_*.pkl and agent_*.json."""
    frames = []

    scene = load_scene_data(ep_dir)
    init_frame = {
            "timestamp": "init",
            "position": scene['agent']['position'],
            "rotation": scene['agent']['rotation'],
        }
    init_frame["visualData"] = init_frame.copy()
    init_frame["visualData"]["detectedObjects"] = []
    frames.append(
        init_frame
    )
    pkl_files = sorted(glob.glob(os.path.join(ep_dir, 'agent_*.pkl')),
                   key=extract_num)

    json_files = sorted(glob.glob(os.path.join(ep_dir, 'agent_*.json')),
                        key=extract_num)

    for fp in pkl_files:
        try:
            with open(fp, 'rb') as f:
                data = _pickle.load(f)
                frames.append(data)
        except Exception as e:
            print(f"[warn] failed to read {fp}: {e}")
    for fp in json_files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                frames.append(_json.load(f))
        except Exception as e:
            print(f"[warn] failed to read {fp}: {e}")
    return frames


def load_frames_any(ep_dir: str) -> List[Dict]:
    """Try human format first, then agent generic patterns."""
    fr = load_frames(ep_dir)
    if fr:
        return fr
    return load_agent_frames_generic(ep_dir)


def frames_to_seconds(frames: List[Dict], agent_mode: bool = False) -> List[float]:
    """Return per-frame elapsed seconds starting at 0.0.
    Uses string 'timestamp' if present; else uses 'step' with a heuristic scale.
    """
    if not frames:
        return []
    # Try timestamps
    ts = []
    for fr in frames:
        t = fr.get('timestamp')
        if isinstance(t, str):
            dt = parse_timestamp(t)
            ts.append(dt)
        else:
            ts = None
            break
    if ts and all(ts):
        t0 = ts[0]
        return [float((t - t0).total_seconds()) for t in ts]
    # Fallback to steps (heuristic seconds per step)
    secs_per_step = 1.0 if agent_mode else (1.0 / 1.31)  # treat agent steps as 1s
    steps = [fr.get('step') for fr in frames]
    if steps and all(isinstance(s, (int, float)) for s in steps):
        s0 = steps[0]
        return [float(s - s0) * secs_per_step for s in steps]
    # Last resort: uniform spacing 1.0s
    return [float(i) for i in range(len(frames))]


def angle_wrap_deg(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return a


def infer_actions(frames: List[Dict], turn_thresh_deg: float = 7.5, move_thresh_m: float = 0.08, agent_mode: bool = False) -> Tuple[List[str], List[float]]:
    """Infer per-step actions and ensure the last frame is always 'commit'.
    Returns (actions, action_timestamps_seconds).
    """
    if not frames:
        return [], []
    times = frames_to_seconds(frames, agent_mode=agent_mode)
    acts = []
    # Infer actions for transitions between frames
    for i in range(len(frames) - 1):
        a = frames[i]
        b = frames[i + 1]
        ax, az = a['position']['x'], a['position']['z']
        bx, bz = b['position']['x'], b['position']['z']
        dyaw = angle_wrap_deg(b['rotation'].get('y', 0.0) - a['rotation'].get('y', 0.0))
        dist = math.hypot(bx - ax, bz - az)
        if abs(dyaw) >= turn_thresh_deg and dist < move_thresh_m:
            acts.append('turn_left' if dyaw > 0 else 'turn_right')
        elif dist >= move_thresh_m:
            acts.append('step_forward')
        else:
            acts.append('stay')
    # Append commit for the last frame
    acts.append('commit')
    # Action timestamps: use the leading frame time for steps, and last-frame time for commit
    lead_times = times[:-1] if len(times) > 1 else times
    act_times = list(lead_times) + [times[-1]]
    return acts, act_times


def plot_actions_episode(pid: str, ep_dir: str, out_dir: str, use_seconds: bool = True, bin_sec: float = 0.0):
    frames = load_frames(ep_dir)
    acts, act_times = infer_actions(frames)
    if not acts:
        print(f"[skip] no frames in {ep_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    y = [ACTION_TO_Y[a] for a in acts]
    if use_seconds:
        x = np.array(act_times)
        xlabel = 'time (s)'
    else:
        x = np.arange(len(acts))
        xlabel = 'frame'

    # Optional aggregation into fixed-width time bins
    if use_seconds and bin_sec and bin_sec > 0:
        t0, t1 = float(np.min(x)), float(np.max(x))
        bins = np.arange(t0, t1 + bin_sec, bin_sec)
        # pick first action occurring in each bin (or most frequent)
        b_centers = []
        b_vals = []
        for b_start, b_end in zip(bins[:-1], bins[1:]):
            mask = (x >= b_start) & (x < b_end)
            if not np.any(mask):
                continue
            vals = [y[i] for i, m in enumerate(mask) if m]
            # choose most frequent action in the bin
            uniq, counts = np.unique(vals, return_counts=True)
            b_vals.append(int(uniq[int(np.argmax(counts))]))
            b_centers.append((b_start + b_end) / 2.0)
        x, y = np.array(b_centers), np.array(b_vals)

    fig, ax = plt.subplots(figsize=(max(8, len(y) * 0.06), 2.8))
    ax.scatter(x, y, c=y, cmap='tab10', s=28)
    ax.set_yticks(list(ACTION_TO_Y.values()))
    ax.set_yticklabels(list(ACTION_TO_Y.keys()))
    ax.set_xlabel(xlabel)
    ax.set_title(f'{pid} — {os.path.basename(ep_dir)} actions')
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    out = os.path.join(out_dir, f'actions_{pid}_{os.path.basename(ep_dir)}.png')
    plt.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[ok] saved {out}")


def aggregate_histograms(eps: List[str], pid: str, out_dir: str, bin_sec: float = 1.0, dur_bucket_sec: float = 10.0):
    """Build stacked action histograms over absolute time for episodes grouped by similar durations.
    Produces one figure per duration bucket and an overall figure for this participant.
    """
    # Collect per-episode per-bin counts
    buckets = {}
    all_counts = None
    all_bins = None
    for ep in eps:
        bins, counts, t_end = _episode_counts(ep, bin_sec)
        if bins is None:
            continue
        key = int((t_end // dur_bucket_sec) * dur_bucket_sec)
        buckets.setdefault(key, []).append((bins, counts))
        # accumulate for overall (align to longest)
        if all_bins is None or bins[-1] > all_bins[-1]:
            all_bins = bins
        if all_counts is None or (counts.shape[1] > all_counts.shape[1]):
            all_counts = np.zeros((len(ACTION_ORDER), len(bins) - 1), dtype=int)
        # add with alignment
        all_counts[:, :counts.shape[1]] += counts

    os.makedirs(out_dir, exist_ok=True)

    def _plot_stacked(bins, counts, title, outfile):
        centers = (bins[:-1] + bins[1:]) / 2.0
        fig, ax = plt.subplots(figsize=(max(10, len(centers) * 0.05), 3.6))
        bottom = np.zeros_like(centers, dtype=float)
        colors = plt.get_cmap('tab10').colors
        for ai, act in enumerate(ACTION_ORDER):
            ax.bar(centers, counts[ai], width=(bins[1]-bins[0])*0.9, bottom=bottom, color=colors[ai % len(colors)], label=act)
            bottom += counts[ai]
        ax.set_xlabel('time (s)')
        ax.set_ylabel('# actions')
        ax.set_title(title)
        ax.legend(ncol=len(ACTION_ORDER), fontsize=8)
        ax.grid(True, axis='y', alpha=0.2)
        plt.tight_layout()
        plt.savefig(outfile, dpi=180)
        plt.close(fig)
        print(f"[ok] saved {outfile}")

    # Per duration bucket
    # for key, items in sorted(buckets.items()):
    #     # Align to the max bins in this bucket
    #     max_bins = max(len(b[0]) for b in items)
    #     bin_ref = None
    #     acc = np.zeros((len(ACTION_ORDER), max_bins - 1), dtype=int)
    #     for bins, c in items:
    #         if bin_ref is None or bins[-1] > bin_ref[-1]:
    #             bin_ref = bins
    #         acc[:, :c.shape[1]] += c
    #     title = f'{pid} — duration ~{key}-{key+dur_bucket_sec:.0f}s (n={len(items)})'
    #     outfile = os.path.join(out_dir, f'agg_{pid}_dur_{int(key)}_{int(key+dur_bucket_sec)}.png')
    #     _plot_stacked(bin_ref, acc, title, outfile)

    # Overall
    if all_bins is not None and all_counts is not None:
        title = f'{pid} — all episodes aggregated'
        outfile = os.path.join(out_dir, f'agg_{pid}_all.png')
        _plot_stacked(all_bins, all_counts, title, outfile)


def _episode_counts(ep: str, bin_sec: float) -> Tuple[np.ndarray, np.ndarray, float]:
    frames = load_frames_any(ep)
    acts, act_times = infer_actions(frames)
    if not acts:
        return None, None, 0.0
    t_end = max(act_times) if act_times else 0.0
    t0, t1 = 0.0, max(t_end, bin_sec)
    bins = np.arange(t0, t1 + bin_sec, bin_sec)
    counts = np.zeros((len(ACTION_ORDER), len(bins) - 1), dtype=int)
    for a, t in zip(acts, act_times):
        placed = False
        for bi in range(len(bins) - 1):
            if t >= bins[bi] and t < bins[bi + 1]:
                counts[ACTION_TO_Y[a], bi] += 1
                placed = True
                break
        if not placed and abs(t - bins[-1]) < 1e-9:
            # Put exact-last-edge events (e.g., commit) into the last bin
            counts[ACTION_TO_Y[a], -1] += 1
    return bins, counts, t_end


def _load_maps_meta(root_maps: str) -> dict:
    try:
        with open(root_maps, 'r', encoding='utf-8') as f:
            return _json.load(f)
    except Exception:
        return {}


def _episode_condition(pid: str, ep_dir: str, maps_meta: dict):
    try:
        idx = int(os.path.basename(ep_dir))
    except Exception:
        return None, None
    meta = (maps_meta or {}).get(pid, [])
    if isinstance(meta, list) and idx < len(meta):
        m = meta[idx]
        map_id = m.get('map_id', m.get('id'))
        cond = m.get('conditions', m.get('condition', {})) or {}
        angle = cond.get('angle', '')
        distractors = cond.get('distractors', '')
        num_cars = cond.get('num_cars', '')
        return map_id, (angle, num_cars, distractors)
    return None, None


def find_agent_episodes(agent_root: str) -> Dict[str, List[str]]:
    eps = {}
    if not agent_root or not os.path.isdir(agent_root):
        return eps
    for name in sorted(os.listdir(agent_root)):
        if not name.startswith('agent_'):
            continue
        base = os.path.join(agent_root, name)
        if not os.path.isdir(base):
            continue
        trials = [os.path.join(base, d) for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
        # keep only those with scene_data_*.json
        trials = [t for t in trials if glob.glob(os.path.join(t, 'scene_data_*.json'))]
        if trials:
            eps[name] = trials
    return eps


def _agent_episode_condition(ep_dir: str, maps_meta: dict):
    """Return (map_id, (angle, num_cars, distractors)) for an agent trial dir."""
    scene_files = glob.glob(os.path.join(ep_dir, 'scene_data_*.json'))
    if not scene_files:
        return None, None
    try:
        with open(scene_files[0], 'r', encoding='utf-8') as f:
            sj = _json.load(f)
        map_id = sj.get('map_id') or (sj.get('agent') or {}).get('map_id')
        cond = sj.get('conditions') or sj.get('condition')
        if not cond and map_id is not None:
            # backfill from maps.json if available
            by_pid = maps_meta or {}
            for v in by_pid.values():
                if isinstance(v, list):
                    for m in v:
                        if m.get('map_id', m.get('id')) == map_id:
                            cond = m.get('conditions', m.get('condition', {}))
                            break
        if map_id is None and cond is None:
            return None, None
        angle = (cond or {}).get('angle', '')
        num_cars = (cond or {}).get('num_cars', '')
        distractors = (cond or {}).get('distractors', '')
        return map_id, (angle, num_cars, distractors)
    except Exception:
        return None, None


def _merge_counts(items: List[Tuple[np.ndarray, np.ndarray]]):
    if not items:
        return None, None
    max_bins_len = max(len(b[0]) for b in items)
    bin_ref = None
    acc = np.zeros((len(ACTION_ORDER), max_bins_len - 1), dtype=int)
    for bins, c in items:
        if bin_ref is None or bins[-1] > bin_ref[-1]:
            bin_ref = bins
        acc[:, :c.shape[1]] += c
    return bin_ref, acc


def _plot_humans_agents(bins_h, counts_h, bins_a, counts_a, title, outfile):
    if bins_h is None and bins_a is None:
        return
    fig, axes = plt.subplots(2 if (bins_h is not None and bins_a is not None) else 1, 1,
                             figsize=(12, 6 if (bins_h is not None and bins_a is not None) else 3.6),
                             squeeze=False)
    row = 0
    def _draw(ax, bins, counts, subtitle):
        centers = (bins[:-1] + bins[1:]) / 2.0
        # Normalize to 100% stacked per-bin
        totals = np.sum(counts, axis=0)
        denom = np.where(totals > 0, totals, 1.0)
        frac = counts / denom  # shape: (actions, nbins)
        bottom = np.zeros_like(centers, dtype=float)
        colors = plt.get_cmap('tab10').colors
        for ai, act in enumerate(ACTION_ORDER):
            ax.bar(centers, frac[ai] * 100.0, width=(bins[1]-bins[0]) * 0.9,
                   bottom=bottom * 100.0, color=colors[ai % len(colors)], label=act)
            bottom += frac[ai]
        ax.set_xlabel('time (s)')
        ax.set_ylabel('% of actions')
        ax.set_ylim(0, 100)
        ax.set_title(subtitle)
        ax.grid(True, axis='y', alpha=0.2)

    if bins_h is not None:
        _draw(axes[row, 0], bins_h, counts_h, f'humans — {title}')
        row += 1
    if bins_a is not None:
        _draw(axes[row, 0], bins_a, counts_a, f'agents — {title}')
    # shared legend
    handles = [plt.Rectangle((0,0),1,1,color=plt.get_cmap('tab10').colors[i%10]) for i,_ in enumerate(ACTION_ORDER)]
    labels = ACTION_ORDER
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    plt.tight_layout(rect=[0,0,0.95,1])
    plt.savefig(outfile, dpi=180)
    plt.close(fig)
    print(f"[ok] saved {outfile}")


def aggregate_all_participants(eps_by_pid: Dict[str, List[str]], out_dir: str, bin_sec: float, dur_bucket_sec: float,
                               agent_root: str = None):
    """Aggregate stacked histograms across all participants (no duration buckets):
    - per map_id
    - per condition group (angle, distractors)
    """
    maps_path = os.path.join(_ROOT, 'maps', 'maps.json')
    maps_meta = _load_maps_meta(maps_path)

    per_map_h = {}           # humans: map_id -> list of (bins, counts)
    per_cond1_h = {}         # humans: independent vars
    per_cond2_h = {}         # humans: two-way interactions
    per_cond3_h = {}         # humans: three-way interaction

    for pid, eps in eps_by_pid.items():
        for ep in eps:
            bins, counts, dur = _episode_counts(ep, bin_sec)
            if bins is None:
                continue
            map_id, cond = _episode_condition(pid, ep, maps_meta)
            if map_id is not None:
                per_map_h.setdefault(map_id, []).append((bins, counts))
            if cond is not None:
                angle = cond[0]
                num_cars = cond[1]
                distractors = cond[2]
                # independent
                per_cond1_h.setdefault(('angle', angle), []).append((bins, counts))
                per_cond1_h.setdefault(('cars', num_cars), []).append((bins, counts))
                per_cond1_h.setdefault(('dist', distractors), []).append((bins, counts))
                # two-way
                per_cond2_h.setdefault(('angle', angle, 'dist', distractors), []).append((bins, counts))
                per_cond2_h.setdefault(('angle', angle, 'cars', num_cars), []).append((bins, counts))
                per_cond2_h.setdefault(('cars', num_cars, 'dist', distractors), []).append((bins, counts))
                # three-way
                per_cond3_h.setdefault((angle, num_cars, distractors), []).append((bins, counts))

    os.makedirs(out_dir, exist_ok=True)

    def _plot_stacked(bins, counts, title, outfile):
        centers = (bins[:-1] + bins[1:]) / 2.0
        fig, ax = plt.subplots(figsize=(max(10, len(centers) * 0.05), 3.8))
        bottom = np.zeros_like(centers, dtype=float)
        colors = plt.get_cmap('tab10').colors
        for ai, act in enumerate(ACTION_ORDER):
            ax.bar(centers, counts[ai], width=(bins[1]-bins[0])*0.9, bottom=bottom, color=colors[ai % len(colors)], label=act)
            bottom += counts[ai]
        ax.set_xlabel('time (s)')
        ax.set_ylabel('# actions')
        ax.set_title(title)
        ax.legend(ncol=len(ACTION_ORDER), fontsize=8)
        ax.grid(True, axis='y', alpha=0.2)
        plt.tight_layout()
        plt.savefig(outfile, dpi=180)
        plt.close(fig)
        print(f"[ok] saved {outfile}")

    # If agent_root provided, collect the same for agents
    per_map_a = {}; per_cond1_a = {}; per_cond2_a = {}; per_cond3_a = {}
    if agent_root:
        agent_eps = find_agent_episodes(agent_root)
        for agent, trials in agent_eps.items():
            for ep in trials:
                bins, counts, _ = _episode_counts(ep, bin_sec)
                if bins is None:
                    continue
                map_id, cond = _agent_episode_condition(ep, maps_meta)
                if map_id is not None:
                    per_map_a.setdefault(map_id, []).append((bins, counts))
                if cond is not None:
                    angle, num_cars, distractors = cond
                    per_cond1_a.setdefault(('angle', angle), []).append((bins, counts))
                    per_cond1_a.setdefault(('cars', num_cars), []).append((bins, counts))
                    per_cond1_a.setdefault(('dist', distractors), []).append((bins, counts))
                    per_cond2_a.setdefault(('angle', angle, 'dist', distractors), []).append((bins, counts))
                    per_cond2_a.setdefault(('angle', angle, 'cars', num_cars), []).append((bins, counts))
                    per_cond2_a.setdefault(('cars', num_cars, 'dist', distractors), []).append((bins, counts))
                    per_cond3_a.setdefault((angle, num_cars, distractors), []).append((bins, counts))

    # Helper to produce combined human/agent plots and filenames
    def _emit_from_dict(h_dict, a_dict, title_prefix, name_fn):
        keys = set(h_dict.keys()) | set(a_dict.keys())
        for k in sorted(keys, key=lambda x: str(x)):
            bins_h, counts_h = _merge_counts(h_dict.get(k, []))
            bins_a, counts_a = _merge_counts(a_dict.get(k, []))
            title = f'{title_prefix} {k}'
            outfile = os.path.join(out_dir, f'{name_fn(k)}.png')
            _plot_humans_agents(bins_h, counts_h, bins_a, counts_a, title, outfile)

    # Per map
    # _emit_from_dict(per_map_h, per_map_a, 'map', lambda k: f'agg_all_map_{k}')

    # Independent vars
    def name1(k):
        var, val = k
        if var == 'angle':
            return f'agg_all_{val}'
        if var == 'cars':
            return f'agg_all_cars{val}'
        return f'agg_all_dist{val}'
    _emit_from_dict(per_cond1_h, per_cond1_a, 'independent', name1)

    # Two-way
    def name2(k):
        a, aval, b, bval = k
        parts = []
        parts.append(aval if a == 'angle' else (f'{a}{aval}'))
        parts.append(bval if b == 'angle' else (f'{b}{bval}'))
        return 'agg_all_' + '_'.join(parts).replace('dist', 'dist')
    _emit_from_dict(per_cond2_h, per_cond2_a, 'two-way', name2)

    # Three-way
    def name3(k):
        angle, cars, dist = k
        return f'agg_all_{angle}_cars{cars}_dist{dist}'
    _emit_from_dict(per_cond3_h, per_cond3_a, 'three-way', name3)


def _add_events(store: Dict[str, List[float]], frames: List[Dict], agent_mode: bool):
    acts, times = infer_actions(frames, agent_mode=agent_mode)
    for a, t in zip(acts, times):
        if a == 'commit':
            continue
        store.setdefault(a, []).append(float(t))


def _cumulative_series(ts: List[float], t_max: float):
    if not ts:
        return [0.0, t_max], [0, 0]
    ts = sorted(ts)
    xs = [0.0]
    ys = [0]
    c = 0
    for t in ts:
        xs.extend([t, t])
        ys.extend([c, c+1])
        c += 1
    xs.append(t_max)
    ys.append(c)
    return xs, ys


def _plot_lines_actions(h_events: Dict[str, List[float]], a_events: Dict[str, List[float]], title: str, outfile: str):
    actions = ['turn_left', 'turn_right', 'step_forward', 'stay']
    t_max = 0.0
    for d in (h_events, a_events):
        for ts in d.values():
            if ts:
                t_max = max(t_max, max(ts))
    t_max = max(t_max, 1.0)

    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8), squeeze=False)
    for i, act in enumerate(actions):
        ax = axs[0, i]
        # Humans: cumulative line (no binning)
        xs_h, ys_h = _cumulative_series(h_events.get(act, []), t_max)
        ax.plot(xs_h, ys_h, color='tab:blue', lw=2, label='humans', drawstyle='steps-post')
        # Agents: per-second bars
        ats = a_events.get(act, []) or []
        if ats:
            bins = np.arange(0.0, math.ceil(t_max) + 1.0, 1.0)
            hist, _ = np.histogram(ats, bins=bins)
            centers = (bins[:-1] + bins[1:]) / 2.0
            ax.bar(centers, hist, width=0.9, color='tab:orange', alpha=0.4, label='agents (per sec)')
        ax.set_title(act)
        ax.set_xlabel('time (s)')
        if i == 0:
            ax.set_ylabel('# actions')
        ax.grid(True, alpha=0.2)
    handles = [plt.Line2D([0],[0], color='tab:blue', lw=2, label='humans'),
               plt.Rectangle((0,0),1,1, color='tab:orange', alpha=0.4, label='agents (per sec)')]
    fig.legend(handles=handles, labels=['humans','agents (per sec)'], loc='upper right')
    fig.suptitle(title)
    plt.tight_layout(rect=[0,0,0.95,0.95])
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=180)
    plt.close(fig)
    print(f"[ok] saved {outfile}")


def aggregate_all_participants_lines(eps_by_pid: Dict[str, List[str]], out_root: str, agent_root: str = None):
    maps_path = os.path.join(_ROOT, 'maps', 'maps.json')
    maps_meta = _load_maps_meta(maps_path)

    # Group containers
    per_map_h = {}
    per_cond1_h = {}
    per_cond2_h = {}
    per_cond3_h = {}

    for pid, eps in eps_by_pid.items():
        for ep in eps:
            frames = load_frames(ep)
            if not frames:
                continue
            map_id, cond = _episode_condition(pid, ep, maps_meta)
            if map_id is not None:
                store = per_map_h.setdefault(map_id, {})
                _add_events(store, frames, agent_mode=False)
            if cond is not None:
                angle, num_cars, distractors = cond
                per_cond1_h.setdefault(('angle', angle), {})
                _add_events(per_cond1_h[('angle', angle)], frames, agent_mode=False)
                per_cond1_h.setdefault(('cars', num_cars), {})
                _add_events(per_cond1_h[('cars', num_cars)], frames, agent_mode=False)
                per_cond1_h.setdefault(('dist', distractors), {})
                _add_events(per_cond1_h[('dist', distractors)], frames, agent_mode=False)
                per_cond2_h.setdefault(('angle', angle, 'dist', distractors), {})
                _add_events(per_cond2_h[('angle', angle, 'dist', distractors)], frames, agent_mode=False)
                per_cond2_h.setdefault(('angle', angle, 'cars', num_cars), {})
                _add_events(per_cond2_h[('angle', angle, 'cars', num_cars)], frames, agent_mode=False)
                per_cond2_h.setdefault(('cars', num_cars, 'dist', distractors), {})
                _add_events(per_cond2_h[('cars', num_cars, 'dist', distractors)], frames, agent_mode=False)
                per_cond3_h.setdefault((angle, num_cars, distractors), {})
                _add_events(per_cond3_h[(angle, num_cars, distractors)], frames, agent_mode=False)

    # Agents
    per_map_a = {}; per_cond1_a = {}; per_cond2_a = {}; per_cond3_a = {}
    if agent_root:
        agent_eps = find_agent_episodes(agent_root)
        for agent, trials in agent_eps.items():
            for ep in trials:
                frames = load_agent_frames_generic(ep)
                if not frames:
                    continue
                map_id, cond = _agent_episode_condition(ep, maps_meta)
                if map_id is not None:
                    store = per_map_a.setdefault(map_id, {})
                    _add_events(store, frames, agent_mode=True)
                if cond is not None:
                    angle, num_cars, distractors = cond
                    per_cond1_a.setdefault(('angle', angle), {})
                    _add_events(per_cond1_a[('angle', angle)], frames, agent_mode=True)
                    per_cond1_a.setdefault(('cars', num_cars), {})
                    _add_events(per_cond1_a[('cars', num_cars)], frames, agent_mode=True)
                    per_cond1_a.setdefault(('dist', distractors), {})
                    _add_events(per_cond1_a[('dist', distractors)], frames, agent_mode=True)
                    per_cond2_a.setdefault(('angle', angle, 'dist', distractors), {})
                    _add_events(per_cond2_a[('angle', angle, 'dist', distractors)], frames, agent_mode=True)
                    per_cond2_a.setdefault(('angle', angle, 'cars', num_cars), {})
                    _add_events(per_cond2_a[('angle', angle, 'cars', num_cars)], frames, agent_mode=True)
                    per_cond2_a.setdefault(('cars', num_cars, 'dist', distractors), {})
                    _add_events(per_cond2_a[('cars', num_cars, 'dist', distractors)], frames, agent_mode=True)
                    per_cond3_a.setdefault((angle, num_cars, distractors), {})
                    _add_events(per_cond3_a[(angle, num_cars, distractors)], frames, agent_mode=True)

    # Output dirs
    base = os.path.join(out_root, 'line')
    d_map = os.path.join(base, 'map')
    d_ind = os.path.join(base, 'independent')
    d_two = os.path.join(base, 'two_way')
    d_thr = os.path.join(base, 'three_way')
    for d in (d_map, d_ind, d_two, d_thr):
        os.makedirs(d, exist_ok=True)

    def _emit(dhum, dagn, folder, name_fn, title_prefix):
        keys = set(dhum.keys()) | set(dagn.keys())
        for k in sorted(keys, key=lambda x: str(x)):
            h_ev = dhum.get(k, {})
            a_ev = dagn.get(k, {})
            title = f'{title_prefix} {k}'
            outfile = os.path.join(folder, f'{name_fn(k)}.png')
            _plot_lines_actions(h_ev, a_ev, title, outfile)

    _emit(per_map_h, per_map_a, d_map, lambda k: f'agg_all_map_{k}', 'map')
    def name1(k):
        var, val = k
        if var == 'angle':
            return f'agg_all_{val}'
        if var == 'cars':
            return f'agg_all_cars{val}'
        return f'agg_all_dist{val}'
    _emit(per_cond1_h, per_cond1_a, d_ind, name1, 'independent')
    def name2(k):
        a, aval, b, bval = k
        parts = []
        parts.append(aval if a == 'angle' else (f'{a}{aval}'))
        parts.append(bval if b == 'angle' else (f'{b}{bval}'))
        return 'agg_all_' + '_'.join(parts).replace('dist', 'dist')
    _emit(per_cond2_h, per_cond2_a, d_two, name2, 'two-way')
    def name3(k):
        angle, cars, dist = k
        return f'agg_all_{angle}_cars{cars}_dist{dist}'
    _emit(per_cond3_h, per_cond3_a, d_thr, name3, 'three-way')


def _find_target(scene_data: Dict):
    if not scene_data or 'vehicles' not in scene_data:
        return None
    for v in scene_data['vehicles']:
        if v.get('isTarget') or 'Target' in v.get('name', ''):
            return v
    return None


def compute_audio_log_like(step: Dict, scene_data: Dict, ep_dir: str) -> np.ndarray:
    """Env-matched ITD audio log-likelihood over (r,theta) using WAV-derived ITD.
    - Extract (itd_seconds, ild_db) from step['audioFileName'] if present; otherwise 0.
    - Build log_audio_like_theta with Gaussian over THETA_GRID and smooth with gaussian_filter1d(mode='wrap').
    - Tile across R_GRID.
    """
    thetas = THETA_GRID
    rs = R_GRID
    head_radius = 0.0875
    c = 343.0
    sigma_itd = 2.0e-5

    # Observed ITD from WAV if available
    itd_obs = 0.0
    wav_name = step.get('audioFileName')
    if wav_name:
        wav_path = os.path.join(ep_dir, wav_name)
        try:
            itd_obs, _ild = compute_itd_ild_from_wav(wav_path)
        except Exception as _e:
            pass

    # Forward model across theta (Woodworth piecewise symmetric)
    # itd_pred = []
    # for th in thetas:
    #     x = abs(float(th))
    #     sgn = 1.0 if th >= 0 else -1.0
    #     if x <= math.pi / 2:
    #         itd_m = (head_radius / c) * (x + math.sin(x))
    #     else:
    #         itd_m = (head_radius / c) * (math.pi - x + math.sin(x))
    #     itd_pred.append(sgn * itd_m)
    # itd_pred = np.array(itd_pred)
    itd_pred = []
    ild_pred = []
    for theta in thetas:
        # Woodworth ITD model (approx, symmetric)
        itd_model = (head_radius / c) * (abs(theta) + math.sin(abs(theta)))
        itd_model *= np.sign(theta)
        itd_pred.append(itd_model)
    itd_pred = np.array(itd_pred)

    # Log-likelihood over theta, smoothed (wrap)
    log_audio_like_theta = -(itd_obs - itd_pred) ** 2 / (2 * sigma_itd ** 2) - np.log(math.sqrt(2 * math.pi) * sigma_itd)
    log_audio_like_theta -= np.max(log_audio_like_theta)
    try:
        log_audio_like_theta = gaussian_filter1d(log_audio_like_theta, sigma=1.0 / (thetas[1] - thetas[0]), mode='wrap')
    except Exception:
        pass
    return np.tile(log_audio_like_theta, (len(rs), 1))


def _ang_diff(a: np.ndarray, b: float) -> np.ndarray:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def compute_visual_log_like(step: Dict, scene_data: Dict) -> np.ndarray:
    """Visual log-likelihood matching the provided env logic more closely.
    - Builds an update_map by aggregating contributions from each detected object.
    - Uses 2x4 car footprint expansion (grid-aligned) and supports sharp peaks mode.
    - Range kernel sigma_r = max(alpha_r * r, sigma_r_min), bearing uses von Mises with kappa ~ 1/sigma_theta^2.
    - Color weighting: full weight for target-color cars, lambda_color for others.
    """
    thetas = THETA_GRID
    rs = R_GRID
    vd = step.get('visualData') or {}
    detected_objects = vd.get('detectedObjects') or []
    if not detected_objects:
        return None


    # Parameters (mirror env defaults where applicable)
    sigma_theta_deg = 2.5
    sigma_theta = np.deg2rad(sigma_theta_deg)
    kappa = 1.0 / max(sigma_theta ** 2, 1e-6)
    alpha_r = 0.01
    sigma_r_min = 0.01
    visible_weight = 0.7
    lambda_color = 0.0

    sharp_visual_peaks = (sigma_theta_deg <= 1e-3) and (sigma_r_min <= 1e-4) and (alpha_r <= 1e-6)
    # sharp_visual_peaks = True

    # Pose
    ax, az = step['position']['x'], step['position']['z']
    ayaw = math.radians(step['rotation'].get('y', 0.0))

    # Target color from scene
    tgt = _find_target(scene_data)
    if tgt is not None:
        tname = tgt.get('name', '')
        if 'Black' in tname:
            target_color = 0
        elif 'White' in tname:
            target_color = 2
        else:
            target_color = 1
    else:
        target_color = None

    def wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def footprint_cells_centered(cx: int, cy: int):
        xs = [cx - 1, cx]
        ys = [cy - 2, cy - 1, cy, cy + 1]
        for xx in xs:
            if xx < 0 or xx >= 1000:  # no strict bound; will be clipped by map extents effectively
                pass
            for yy in ys:
                yield xx, yy

    update_map = np.zeros((len(rs), len(thetas)), dtype=float)
    visible = 0
    for det in detected_objects:
        # Center in grid coords using world position
        pos = det.get('position') or {}
        if 'x' not in pos or 'z' not in pos:
            continue
        cx_f, cy_f = unity_to_grid(float(pos['x']), float(pos['z']))
        cx, cy = int(round(cx_f)), int(round(cy_f))
        # Color index
        name = det.get('name', '')
        if 'Black' in name:
            obj_color = 0
        elif 'White' in name:
            obj_color = 2
        else:
            obj_color = 1
        color_weight = visible_weight if (target_color is None or obj_color == target_color) else lambda_color

        if sharp_visual_peaks:
            # Concentrate directly at nearest (r,theta) cell for each footprint cell
            for xx, yy in footprint_cells_centered(cx, cy):
                dx = float(xx) - float(cx_f) + (cx_f - ax)  # approximate world→grid offset
                dy = float(yy) - float(cy_f) + (cy_f - az)
                # But better: compute from agent position in world to grid cell center
                # Agent in grid
                agx, agy = unity_to_grid(ax, az)
                dx = float(xx) - float(agx)
                dy = float(yy) - float(agy)
                r_cell = math.hypot(dx, dy)
                theta_abs = math.atan2(dx, -dy) % (2 * math.pi)
                theta_cell = wrap_angle(theta_abs - ayaw)
                i_r = int(np.argmin(np.abs(rs - r_cell)))
                theta_diffs = wrap_angle(thetas - theta_cell)
                i_th = int(np.argmin(np.abs(theta_diffs)))
                if 0 <= i_r < len(rs) and 0 <= i_th < len(thetas):
                    update_map[i_r, i_th] += color_weight
            visible += 1
            continue

        # Soft kernel accumulated over 2x4 footprint
        contrib = np.zeros_like(update_map)
        n_cells = 0
        # Agent in grid for dx,dy
        agx, agy = unity_to_grid(ax, az)
        for xx, yy in footprint_cells_centered(cx, cy):
            dx = float(xx) - float(agx)
            dy = float(yy) - float(agy)
            r_cell = math.hypot(dx, dy)
            theta_abs = math.atan2(dx, -dy) % (2 * math.pi)
            theta_cell = wrap_angle(theta_abs - ayaw)
            sigma_r = max(alpha_r * r_cell, sigma_r_min)
            range_term = np.exp(-((rs - r_cell) ** 2) / (2.0 * sigma_r ** 2))
            theta_diff = wrap_angle(thetas - theta_cell)
            bearing_term = np.exp(kappa * np.cos(theta_diff))
            contrib += np.outer(range_term, bearing_term)
            n_cells += 1
        if n_cells > 0:
            contrib /= float(n_cells)
        update_map += color_weight * contrib
        visible += 1

    if visible <= 0:
        return None
    update_map /= float(visible)
    update_map /= (np.sum(update_map) + 1e-12)
    return np.log(update_map + 1e-12)


def belief_for_episode(pid: str, ep_dir: str, out_dir: str, step_idx: str = 'last'):
    scene = load_scene_data(ep_dir)
    if scene is None:
        print(f"[skip] no scene for {ep_dir}")
        return
    frames = load_frames(ep_dir)
    if not frames:
        print(f"[skip] no frames in {ep_dir}")
        return
    if step_idx == 'last':
        k = len(frames) - 1
    else:
        k = max(0, min(int(step_idx), len(frames) - 1))
    step = frames[k]
    # Compute audio and visual beliefs; combine as posterior (sum in log-space)
    log_audio = compute_audio_log_like(step, scene, ep_dir)
    log_visual = compute_visual_log_like(step, scene)
    step = dict(step)  # shallow copy
    if log_audio is not None:
        step['log_audio'] = log_audio
    if log_visual is not None:
        step['log_visual'] = log_visual
    if (log_audio is not None) and (log_visual is not None):
        # Simple fusion with equal weights in log-space
        # Normalize both before combining to prevent dominance by scale
        la = log_audio - np.max(log_audio)
        lv = log_visual - np.max(log_visual)
        step['log_posterior'] = la + lv
    elif log_audio is not None:
        step['log_posterior'] = log_audio
    elif log_visual is not None:
        step['log_posterior'] = log_visual
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'beliefs_{pid}_{os.path.basename(ep_dir)}.png')
    save_belief_grid_for_step(step, out, scene_data=scene, trajectory=frames, step_idx=k, pid_label=pid)
    print(f"[ok] saved {out}")


def find_human_episodes(root_dirs: List[str]) -> Dict[str, List[str]]:
    result = {}
    for root in root_dirs:
        for pid in [f'p{i:02d}' for i in range(1, 13)]:
            base = os.path.join(root, pid)
            if not os.path.isdir(base):
                continue
            eps = [d for d in sorted(glob.glob(os.path.join(base, '*'))) if os.path.isdir(d)]
            if eps:
                result.setdefault(pid, []).extend(eps)
    return result


def main():
    ap = argparse.ArgumentParser(description='Human actions timeline and belief overlays')
    ap.add_argument('--root', type=str, default=None, help='Data root (default: auto search analysis/data then ./data)')
    ap.add_argument('--pids', type=str, default='all', help='Comma list of pids (p01..p12) or all')
    ap.add_argument('--limit', type=int, default=5, help='Limit episodes per pid (default 5)')
    ap.add_argument('--do-actions', action='store_true', help='Generate per-episode action timelines')
    ap.add_argument('--do-beliefs', action='store_true', help='Generate world+belief overlays')
    ap.add_argument('--time-seconds', action='store_true', help='Use seconds on x-axis (if timestamps/steps available)')
    ap.add_argument('--bin-sec', type=float, default=0.0, help='Aggregate actions into time bins (seconds)')
    ap.add_argument('--do-aggregate', action='store_true', help='Produce stacked histograms aggregated over episodes per participant and overall')
    ap.add_argument('--dur-bucket', type=float, default=10.0, help='Duration bucket size in seconds for aggregation')
    ap.add_argument('--agent-root', type=str, default=None, help='Root path containing agent_* folders for agent aggregation subplots')
    args = ap.parse_args()

    roots = [args.root] if args.root else [os.path.join('analysis', 'data'), 'data']
    roots = [r for r in roots if r and os.path.isdir(r)]
    if not roots:
        print('[error] no data roots found. Use --root to specify the path containing p01..p12')
        return

    eps_by_pid = find_human_episodes(roots)
    if args.pids != 'all':
        keep = set([p.strip() for p in args.pids.split(',')])
        eps_by_pid = {p: eps for p, eps in eps_by_pid.items() if p in keep}

    out_actions = os.path.join('analysis', 'plots', 'human_actions')
    out_beliefs = os.path.join('analysis', 'plots', 'human_beliefs')

    for pid, eps in eps_by_pid.items():
        selected_eps = eps[: max(0, args.limit)]
        for ep in selected_eps:
            if args.do_actions:
                plot_actions_episode(pid, ep, out_actions, use_seconds=args.time_seconds, bin_sec=args.bin_sec)
            if args.do_beliefs:
                belief_for_episode(pid, ep, out_beliefs, step_idx='last')
        if args.do_aggregate:
            aggregate_histograms(selected_eps, pid, out_actions, bin_sec=max(args.bin_sec, 1.0), dur_bucket_sec=args.dur_bucket)

    # Cross-participant aggregation per map and per condition
    if args.do_aggregate:
        aggregate_all_participants(eps_by_pid, out_actions, bin_sec=max(args.bin_sec, 1.0), dur_bucket_sec=args.dur_bucket,
                                   agent_root=args.agent_root)
        # Also produce line plots with humans vs agents per condition
        aggregate_all_participants_lines(eps_by_pid, out_actions, agent_root=args.agent_root)


if __name__ == '__main__':
    main()
