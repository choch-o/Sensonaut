#!/usr/bin/env python3
import argparse
import concurrent.futures as futures
import json
import os
import re
import sys
import time
import glob
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.metrics import (
    mse_discrepancy,
    dist_discrepancy,
    accuracy_discrepancy,
    composite_discrepancy,
    level_similarity_metrics,
    pattern_similarity_metrics,
    composite_score,
)


def find_models(models_dir: str) -> list[str]:
    paths = []
    for name in os.listdir(models_dir):
        if name.startswith('.'):
            continue
        p = os.path.join(models_dir, name)
        if os.path.isfile(p):
            # SB3 saves as .zip typically; in this repo files may lack extension
            paths.append(p)
    paths.sort()
    return paths


def _parse_bool(s: str) -> bool | None:
    sl = s.strip().lower()
    if sl in ("true", "t", "1", "yes", "y"): return True
    if sl in ("false", "f", "0", "no", "n"): return False
    return None


def _parse_numeric(s: str) -> float | int | None:
    try:
        v = float(s)
        if abs(v - int(v)) < 1e-9:
            return int(v)
        return v
    except Exception:
        return None


def _extract_env_overrides_from_name(model_path: str) -> dict:
    """Extract environment.* overrides from model filename suffix after '_gs_'."""
    base = os.path.basename(model_path)
    env_overrides = {}
    if '_gs_' not in base:
        return env_overrides
    suffix = base.split('_gs_', 1)[1]
    # tokens like: turn-30, fp-0.3, tp-0.1, alpha-0.8, visual_a-0.7, correct_-False
    tokens = [t for t in suffix.split('_') if '-' in t]
    for tok in tokens:
        k, v = tok.split('-', 1)
        k = k.strip()
        v = v.strip()
        key_map = {
            'turn': 'environment.turn_angle',
            'turn_angle': 'environment.turn_angle',
            'fp': 'environment.forward_penalty',
            'forward_penalty': 'environment.forward_penalty',
            'tp': 'environment.turn_penalty',
            'turn_penalty': 'environment.turn_penalty',
            'alpha': 'environment.alpha',
            'visual_a': 'environment.visual_alpha',
            'visual_alpha': 'environment.visual_alpha',
            'correct': 'environment.correct_object',
            'correct_': 'environment.correct_object',
        }
        dest = key_map.get(k)
        if not dest:
            continue
        if dest.endswith('correct_object'):
            bv = _parse_bool(v)
            if bv is not None:
                env_overrides[dest] = bv
        else:
            nv = _parse_numeric(v)
            if nv is not None:
                env_overrides[dest] = nv
    return env_overrides


def _json_overrides_for_model(model_path: str, use_unity: bool, save_videos: bool) -> dict:
    overrides = {
        # Environment/test toggles
        "mode": "test",  # becomes environment.mode
        # Explicitly set use_unity to requested value
        "environment.use_unity": bool(use_unity),
        # Model path
        "model.pretrained_model": model_path,
        # Keep logs light unless requested
        "logging.save_videos": bool(save_videos),
    }
    # Merge in filename-derived environment parameter overrides
    overrides.update(_extract_env_overrides_from_name(model_path))
    return overrides


def _safe_suffix(name: str) -> str:
    # Keep alnum, dash, underscore only
    base = os.path.basename(name)
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", base)


def _latest_test_dir(test_log_base: str) -> str | None:
    if not os.path.isdir(test_log_base):
        return None
    cands = [d for d in os.listdir(test_log_base) if d.startswith('test_')]
    if not cands:
        return None
    cands.sort()
    return os.path.join(test_log_base, cands[-1])


def _count_agent_runs(test_root: str) -> int:
    if not test_root or not os.path.isdir(test_root):
        return 0
    return sum(1 for d in os.listdir(test_root) if d.startswith('agent_') and os.path.isdir(os.path.join(test_root, d)))


def _terminate(proc):
    try:
        proc.terminate()
    except Exception:
        pass


def launch_tests_for_model(model_path: str, config_path: str, test_logs_dir: str, use_unity: bool, save_videos: bool,
                           runs_per_model: int, poll_sec: float, timeout_min: float) -> dict:
    """Launch a single agent.py test process for a given model and stop when enough runs are produced.

    Returns a dict with keys: model, suffix, test_root (path), ok (bool), error (optional)
    """
    exe = sys.executable
    agent_script = str(Path(__file__).resolve().parents[1] / 'agent.py')
    suffix = _safe_suffix(os.path.basename(model_path))

    overrides = _json_overrides_for_model(model_path, use_unity=use_unity, save_videos=save_videos)
    cmd = [
        exe, agent_script,
        '--single',
        '--config', config_path,
        '--overrides-json', json.dumps(overrides),
        '--suffix', suffix,
    ]

    # Compute where this run will write
    # agent._prepare_logging_dirs appends suffix under logging.test_log_dir
    test_log_base = os.path.join(test_logs_dir, suffix)
    Path(test_log_base).mkdir(parents=True, exist_ok=True)

    proc = None
    try:
        proc = __import__('subprocess').Popen(cmd)
    except Exception as e:
        return {"model": model_path, "suffix": suffix, "ok": False, "error": f"failed to start: {e}"}

    # Poll for new test_<timestamp>/ and agent_N directories
    start = time.time()
    test_root = None
    try:
        while True:
            # Discover latest test dir for this suffix
            test_root = _latest_test_dir(test_log_base)
            if test_root and _count_agent_runs(test_root) >= runs_per_model:
                _terminate(proc)
                break
            # Check timeout
            if (time.time() - start) > (timeout_min * 60.0):
                _terminate(proc)
                return {"model": model_path, "suffix": suffix, "test_root": test_root, "ok": False, "error": "timeout"}
            # Check if process already exited (e.g., error)
            ret = proc.poll()
            if ret is not None:
                # Process ended; keep whatever was written
                ok = (ret == 0)
                return {"model": model_path, "suffix": suffix, "test_root": test_root, "ok": ok, "error": None if ok else f"returncode={ret}"}
            time.sleep(poll_sec)
    finally:
        _terminate(proc)

    return {"model": model_path, "suffix": suffix, "test_root": test_root, "ok": True}


def _load_maps_conditions(maps_path: str) -> dict:
    with open(maps_path, 'r', encoding='utf-8') as f:
        maps = json.load(f)
    mapping = {}
    for pid, lst in (maps or {}).items():
        if not isinstance(lst, list):
            continue
        for item in lst:
            if not isinstance(item, dict):
                continue
            mid = item.get('map_id', item.get('id'))
            if mid is not None and mid not in mapping:
                mapping[mid] = item.get('conditions', item.get('condition', {})) or {}
    return mapping


def _parse_timestamp(ts: str):
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            pass
    return None


def _get_trajectory(ep_dir: str) -> list[dict]:
    # Prefer pickled step snapshots if present
    agent_files = sorted(glob.glob(os.path.join(ep_dir, 'agent_*.pkl')))
    traj = []
    for fp in agent_files:
        try:
            with open(fp, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'step' in data and 'position' in data and 'rotation' in data:
                traj.append({
                    'step': int(data.get('step', 0)),
                    'position': data.get('position', {}),
                    'rotation': data.get('rotation', {}),
                })
        except Exception:
            continue
    if traj:
        traj.sort(key=lambda d: d['step'])
        return traj
    # Fallback: JSON step logs, if any
    agent_jsons = sorted(glob.glob(os.path.join(ep_dir, 'agent_*.json')))
    for fp in agent_jsons:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ts = _parse_timestamp(data.get('timestamp', ''))
            if ts is not None:
                traj.append({
                    'timestamp': ts,
                    'position': data.get('position', {}),
                    'rotation': data.get('rotation', {}),
                })
        except Exception:
            continue
    if traj and 'timestamp' in traj[0]:
        traj.sort(key=lambda d: d['timestamp'])
    return traj


def _compute_episode_metrics(traj: list[dict], scene: dict | None, estimate: dict | None) -> dict | None:
    if not traj:
        return None
    start = traj[0]['position']
    end = traj[-1]['position']
    # Distance (Unity x,z)
    dist = float(np.hypot(end.get('x', 0.0) - start.get('x', 0.0), end.get('z', 0.0) - start.get('z', 0.0)))

    # Duration (replicate compare_traj.py logic)
    if 'timestamp' in traj[0]:
        duration = (traj[-1]['timestamp'] - traj[0]['timestamp']).total_seconds()
    else:
        duration = float(traj[-1].get('step', 0)) / 1.31
    speed = dist / duration if duration > 0 else 0.0

    # Accumulated head turns
    acc_turns = 0.0
    for i in range(1, len(traj)):
        prev = float(traj[i-1]['rotation'].get('y', 0.0))
        curr = float(traj[i]['rotation'].get('y', 0.0))
        d = curr - prev
        while d > 180:
            d -= 360
        while d < -180:
            d += 360
        acc_turns += abs(d)

    end_to_target = None
    est_correct = None
    est_to_target = None
    target = None
    if scene and isinstance(scene, dict):
        for v in scene.get('vehicles', []) or []:
            if v.get('isTarget') or ('Target' in str(v.get('name', ''))):
                target = v
                break
    if target:
        tx, tz = float(target['position']['x']), float(target['position']['z'])
        end_to_target = float(np.hypot(end.get('x', 0.0) - tx, end.get('z', 0.0) - tz))

    if estimate and target:
        # For agent logs, estimate contains 'est_world_position'
        if isinstance(estimate, dict) and 'est_world_position' in estimate:
            epos = estimate['est_world_position']
        else:
            epos = estimate
        ex, ez = float(epos.get('x', 0.0)), float(epos.get('z', 0.0))
        est_to_target = float(np.hypot(ex - tx, ez - tz))
        # closest vehicle to estimate == target ?
        best_d = None
        best_is_target = False
        for v in scene.get('vehicles', []) or []:
            vx, vz = float(v['position']['x']), float(v['position']['z'])
            d2 = (vx - ex)**2 + (vz - ez)**2
            if best_d is None or d2 < best_d:
                best_d = d2
                best_is_target = bool(v.get('isTarget') or ('Target' in str(v.get('name',''))))
        est_correct = 1 if best_is_target else 0

    return {
        'distance': dist,
        'duration': duration,
        'speed': speed,
        'accumulated_head_turns': acc_turns,
        'end_x': float(end.get('x', 0.0)),
        'end_z': float(end.get('z', 0.0)),
        'end_to_target_dist': end_to_target,
        'estimate_correct': 0 if est_correct in (None, False) else int(est_correct),
        'estimate_to_target_dist': est_to_target,
    }


def build_model_df(test_root: str, model_name: str, maps_path: str) -> pd.DataFrame:
    """Parse test_logs for one model and return a DataFrame compatible with human CSV."""
    # Load map_id -> conditions mapping
    mapid_to_cond = _load_maps_conditions(maps_path)

    rows = []
    # agent_* directories
    for agent_dir in sorted(glob.glob(os.path.join(test_root, 'agent_*'))):
        participant = os.path.basename(agent_dir)
        # Episode folders are numeric
        for ep_dir in sorted(glob.glob(os.path.join(agent_dir, '*'))):
            if not os.path.isdir(ep_dir):
                continue
            # Load scene and estimate
            scene_files = sorted(glob.glob(os.path.join(ep_dir, 'scene_data_*.json')))
            est_files = sorted(glob.glob(os.path.join(ep_dir, 'estimate_*.json')))
            if not scene_files:
                continue
            try:
                with open(scene_files[0], 'r', encoding='utf-8') as f:
                    scene = json.load(f)
            except Exception:
                continue
            estimate = None
            if est_files:
                try:
                    with open(est_files[0], 'r', encoding='utf-8') as f:
                        estimate = json.load(f)
                except Exception:
                    pass

            traj = _get_trajectory(ep_dir)
            m = _compute_episode_metrics(traj, scene, estimate)
            if not m:
                continue

            map_id = scene.get('map_id')
            if map_id is None and isinstance(scene.get('agent'), dict):
                map_id = scene['agent'].get('map_id')
            cond = mapid_to_cond.get(map_id, {})

            rows.append({
                'map_id': map_id,
                'participant': participant,
                'agent_type': model_name,
                'distance': m['distance'],
                'duration': m['duration'],
                'speed': m['speed'],
                'angle': cond.get('angle'),
                'num_cars': cond.get('num_cars'),
                'distractors': cond.get('distractors'),
                'accumulated_head_turns': m['accumulated_head_turns'],
                'end_x': m['end_x'],
                'end_z': m['end_z'],
                'end_to_target_dist': m['end_to_target_dist'],
                'estimate_correct': m['estimate_correct'],
                'estimate_to_target_dist': m['estimate_to_target_dist'],
                'idx': os.path.basename(ep_dir),
            })

    if not rows:
        return pd.DataFrame(columns=['map_id','participant','agent_type','distance','duration','speed','angle','num_cars','distractors','accumulated_head_turns','end_x','end_z','end_to_target_dist','estimate_correct','estimate_to_target_dist','idx'])

    df = pd.DataFrame(rows)
    # Drop rows missing essential grouping keys; keep only conditions present in humans later via join
    return df


def load_human_df(human_csv: str) -> pd.DataFrame:
    df = pd.read_csv(human_csv)
    # Keep humans only
    if 'agent_type' in df.columns:
        df = df[df['agent_type'] == 'human']
    return df


def compute_summary_for_model(model_df: pd.DataFrame, human_df: pd.DataFrame, model_name: str) -> list[dict]:
    results = []

    def summarize(split_name: str, mdf: pd.DataFrame, hdf: pd.DataFrame):
        # Align to rows with non-null group keys
        mdf = mdf.dropna(subset=['angle','distractors','num_cars'])
        hdf = hdf.dropna(subset=['angle','distractors','num_cars'])
        dep_vars = ['duration','distance','accumulated_head_turns']

        # MSE across condition means
        mse = mse_discrepancy(hdf, mdf, dep_vars)

        # Distances (distributional) per dep var
        dists = {}
        for v in dep_vars:
            d = dist_discrepancy(hdf[[v]].rename(columns={v: v}).dropna(), mdf[[v]].rename(columns={v: v}).dropna(), v)
            dists[v] = d

        acc_diff = accuracy_discrepancy(hdf, mdf)
        comp = composite_discrepancy(hdf, mdf)

        # Extended metrics (level + pattern) and composite score_v2
        ext_lvl = level_similarity_metrics(hdf, mdf)
        ext_pat = pattern_similarity_metrics(hdf, mdf)
        ext_metrics = {**ext_lvl, **ext_pat}
        comp_v2 = composite_score(ext_metrics)

        row = {
            'model': model_name,
            'split': split_name,
            'n_model': int(len(mdf)),
            'n_human': int(len(hdf)),
            'mse_duration': mse.get('duration'),
            'mse_distance': mse.get('distance'),
            'mse_accumulated_head_turns': mse.get('accumulated_head_turns'),
            'mse_overall': mse.get('overall'),
            'acc_diff': acc_diff,
            'composite': comp,
            'wd_duration': dists['duration']['wasserstein'],
            'kl_duration': dists['duration']['kl'],
            'wd_distance': dists['distance']['wasserstein'],
            'kl_distance': dists['distance']['kl'],
            'wd_head_turns': dists['accumulated_head_turns']['wasserstein'],
            'kl_head_turns': dists['accumulated_head_turns']['kl'],
        }
        # Flatten extended metrics into row (prefix m2_ and replace dots)
        for k, v in ext_metrics.items():
            row[f"m2_{k.replace('.', '_')}"] = v
        row["composite_v2"] = comp_v2

        results.append(row)

    # Overall
    summarize('overall', model_df, human_df)
    # Success-only
    summarize('success', model_df[model_df['estimate_correct'] == 1], human_df[human_df['estimate_correct'] == 1])
    # Failure-only
    summarize('failure', model_df[model_df['estimate_correct'] == 0], human_df[human_df['estimate_correct'] == 0])

    return results


def main():
    ap = argparse.ArgumentParser(description='Run models in parallel for testing and analyze logs vs. humans.')
    ap.add_argument('--config', default='configs/config_unity.yaml', help='Config YAML path')
    ap.add_argument('--models-dir', default='models/test_20250909_195443-turn30-alpha0.5-timebonus', help='Directory containing candidate models to test')
    ap.add_argument('--models', nargs='*', default=None, help='Specific model file paths to test (overrides --models-dir)')
    ap.add_argument('--max-procs', type=int, default=4, help='Max parallel test processes')
    ap.add_argument('--runs-per-model', type=int, default=12, help='Number of agent_* runs to collect per model')
    ap.add_argument('--poll-sec', type=float, default=10.0, help='Polling interval (seconds) for log progress')
    ap.add_argument('--timeout-min', type=float, default=120.0, help='Per-model timeout in minutes')
    ap.add_argument('--use-unity', action='store_true', help='Enable Unity client during tests (default: off)')
    ap.add_argument('--save-videos', action='store_true', help='Save videos during tests (default: off)')
    ap.add_argument('--dry-run', action='store_true', help='Do not launch tests; only analyze existing logs if present')
    # ap.add_argument('--out-csv', default='analysis/model_summary/models_vs_humans_summary.csv', help='Output CSV for summary')
    ap.add_argument('--human-csv', default='analysis/df_human.csv', help='CSV with human metrics by map')
    ap.add_argument('--maps-json', default='maps/maps.json', help='maps.json for conditions lookup')
    ap.add_argument('--model-name', required=True, help='Model name that corresponds to directory containing test logs')
    args = ap.parse_args()

    
    # # Discover models
    # if args.models:
    #     models = [m for m in args.models if os.path.exists(m)]
    # else:
    #     models = find_models(args.models_dir)

    # if not models:
    #     print('No models found to test.')
    #     sys.exit(1)

    # Where base test logs live (from config)
    # We parse YAML lightly: logging.test_log_dir default is test_logs
    # To avoid YAML dep here, assume default path as in configs/config_unity.yaml
    test_logs_dir = 'test_logs'

    # Launch tests in parallel (unless dry-run)
    test_infos = []
    if not args.dry_run:
        def run_one(m):
            return launch_tests_for_model(
                model_path=m,
                config_path=args.config,
                test_logs_dir=test_logs_dir,
                use_unity=args.use_unity,
                save_videos=args.save_videos,
                runs_per_model=args.runs_per_model,
                poll_sec=args.poll_sec,
                timeout_min=args.timeout_min,
            )

        with futures.ThreadPoolExecutor(max_workers=max(1, args.max_procs)) as pool:
            futs = {pool.submit(run_one, m): m for m in models}
            for fut in futures.as_completed(futs):
                info = fut.result()
                test_infos.append(info)
                status = 'OK' if info.get('ok') else f"ERR({info.get('error')})"
                print(f"[{status}] {os.path.basename(info.get('model',''))} -> {info.get('test_root')}")
    else:
        # Populate info from existing logs (latest per suffix)
        # for m in models:
        suffix = args.model_name
            # suffix = _safe_suffix(os.path.basename(m))
        test_root = _latest_test_dir(os.path.join(test_logs_dir, suffix))
        test_infos.append({"model": suffix, "suffix": suffix, "test_root": test_root, "ok": bool(test_root)})

    # Load human data
    human_df = load_human_df(args.human_csv)

    # Analyze per model
    all_rows = []
    for info in test_infos:
        model_path = info['model']
        # model_name = _safe_suffix(os.path.basename(model_path))
        model_name = args.model_name
        test_root = info.get('test_root')
        test_root = os.path.join(test_logs_dir, model_name)
        if not test_root or not os.path.isdir(test_root):
            print(test_root)
            print(f"No test logs found for {model_name}; skipping analysis.")
            continue
        mdf = build_model_df(test_root, model_name=model_name, maps_path=args.maps_json)
        if mdf.empty:
            print(f"Empty model DataFrame for {model_name} at {test_root}")
            continue
        # Save the per-model DataFrame (mdf) before summarization for inspection/debugging
        try:
            mdf_out_dir = f'analysis/model-{model_name}' #args.out_csv) or '.'
            os.makedirs(mdf_out_dir, exist_ok=True)
            mdf_out_path = os.path.join(mdf_out_dir, f"mdf_{model_name}.csv")
            mdf.to_csv(mdf_out_path, index=False)
            print(f"Saved mdf for {model_name}: {mdf_out_path}")
        except Exception as e:
            print(f"Warning: failed to save mdf for {model_name}: {e}")
        rows = compute_summary_for_model(mdf, human_df, model_name=model_name)
        all_rows.extend(rows)

    if not all_rows:
        print('No results to write.')
        return

    out_dir = f'analysis/model-{model_name}' or '.'
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(f'{out_dir}/models_vs_humans_summary.csv', index=False)
    print(f"Wrote summary: {f'{out_dir}/models_vs_humans_summary.csv'}")


if __name__ == '__main__':
    main()
