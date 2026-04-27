
import argparse
import concurrent.futures as futures
import copy
import itertools
import json
import math
import os
import subprocess
import sys
from datetime import datetime

import yaml
from agents.ppo_agent import train_agent, test_agent


def _normalize_grid_keys(explicit_grid: dict) -> dict[str, list]:
    """Normalize grid_search keys to dot-paths.

    - If key contains a dot, leave as-is (assumed full path)
    - Else, assume key is under `environment` section
    """
    norm: dict[str, list] = {}
    for k, v in (explicit_grid or {}).items():
        path = k if "." in k else f"environment.{k}"
        norm[path] = list(v)
    return norm


def _infer_values(current):
    """Infer a simple 2-3 point grid around a numeric value."""
    try:
        if isinstance(current, bool):
            return [current]
        if isinstance(current, int):
            # Choose up to 3 integer values around current (avoid <=0)
            a = max(1, int(round(current / 3)))
            b = max(1, int(round(current / 2)))
            c = int(current)
            vals = sorted(set([a, b, c]))
            return vals
        if isinstance(current, float):
            if current <= 0:
                return [current]
            if current <= 1.0:
                a = max(0.0, current / 2)
                b = current
                c = min(1.0, current * 1.5)
            else:
                a = current / 2
                b = current
                c = current * 1.5
            vals = [round(x, 4) for x in sorted(set([a, b, c]))]
            return vals
    except Exception:
        pass
    return [current]


def _get_by_dot_path(d: dict, path: str):
    cur = d
    for part in path.split('.'):
        cur = cur[part]
    return cur


def _set_by_dot_path(d: dict, path: str, value):
    parts = path.split('.')
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _short_name(param_path: str) -> str:
    # Use last token, shorten common names
    name = param_path.split('.')[-1]
    mapping = {
        "turn_angle": "turn",
        "forward_penalty": "fp",
        "turn_penalty": "tp",
        "alpha": "alpha",
    }
    return mapping.get(name, name[:8])


def _make_suffix(combo: dict[str, object]) -> str:
    parts = []
    for k, v in combo.items():
        key = _short_name(k)
        if isinstance(v, float):
            val = f"{v:.3g}"
        else:
            val = str(v)
        parts.append(f"{key}-{val}")
    return "gs_" + "_".join(parts)


def _parse_suffix(s: str) -> dict[str, object] | None:
    """Parse a gs_... suffix back into {short_key: value}.

    Accepts strings like:
      - "gs_turn-45_fp-0.3_tp-0.1_alpha-0.8"
    Returns a dict mapping short names (e.g., 'turn', 'fp') to typed values.
    """
    if not s:
        return None
    # Find the last occurrence of 'gs_' to be robust to prefixes
    idx = s.rfind("gs_")
    if idx < 0:
        return None
    payload = s[idx + 3 :]
    if not payload:
        return None
    items = payload.split("_")
    out: dict[str, object] = {}
    for item in items:
        if "-" not in item:
            continue
        k, v = item.split("-", 1)
        # Try to cast to bool/int/float, else keep string
        vv: object = v
        if v.lower() in ("true", "false"):
            vv = (v.lower() == "true")
        else:
            try:
                if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                    vv = int(v)
                else:
                    vv = float(v)
            except Exception:
                vv = v
        out[k] = vv
    return out if out else None


def _prepare_logging_dirs(cfg: dict, suffix: str) -> dict:
    cfg = copy.deepcopy(cfg)
    log_cfg = cfg.get("logging", {})
    def add_suffix(path: str | None, default: str) -> str:
        base = path or default
        return os.path.join(base, suffix)

    log_cfg["video_dir"] = add_suffix(log_cfg.get("video_dir"), "videos")
    log_cfg["tensorboard_log_dir"] = add_suffix(log_cfg.get("tensorboard_log_dir"), "runs")
    log_cfg["test_log_dir"] = add_suffix(log_cfg.get("test_log_dir"), "test_logs")
    log_cfg["checkpoint_dir"] = add_suffix(log_cfg.get("checkpoint_dir"), "checkpoints")
    # Expose suffix for naming saved models
    log_cfg["name_suffix"] = suffix
    cfg["logging"] = log_cfg
    # Ensure directories exist to avoid collisions later
    for k in ["video_dir", "tensorboard_log_dir", "test_log_dir", "checkpoint_dir"]:
        try:
            os.makedirs(log_cfg[k], exist_ok=True)
        except Exception:
            pass
    return cfg


def _iter_leaf_paths(d: dict, prefix: str = "", allow_sections=("environment", "model", "noise", "physics")):
    """Yield (dot_path, value) for scalar leaves in selected sections.

    Only includes simple scalar types to avoid complex structures in overrides.
    """
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if prefix == "" and k not in allow_sections:
            # only traverse known sections from the root
            continue
        if isinstance(v, dict):
            yield from _iter_leaf_paths(v, path, allow_sections)
        else:
            if isinstance(v, (str, int, float, bool)) or v is None:
                yield (path, v)


def _run_single(config: dict):
    mode = config["environment"]["mode"]
    if mode == "train":
        train_agent(config)
    else:
        test_agent(config)


def main():
    parser = argparse.ArgumentParser(description="Run agent, schedule grid search, execute a single job, or resume from pretrained models.")
    parser.add_argument("--config", default="configs/config_RR.yaml", help="Path to YAML config")
    # Modes
    parser.add_argument("--grid", action="store_true", help="Schedule grid search via grid_search dict in config")
    parser.add_argument("--single", action="store_true", help="Run a single job, applying --overrides-json and optional --suffix")
    parser.add_argument("--grid-from-models", action="store_true", help="Resume grid search from pretrained models (parse suffix to set params)")
    # Single-run options
    parser.add_argument("--overrides-json", type=str, default=None, help="JSON dict of dot-path overrides for single run")
    parser.add_argument("--suffix", type=str, default=None, help="Optional log dir suffix for single run")
    # From-models options
    parser.add_argument("--models", nargs="*", default=None, help="List of pretrained model file paths to resume from")
    parser.add_argument("--models-file", type=str, default=None, help="Text file with one pretrained model path per line")
    parser.add_argument("--resume-mode", type=str, choices=["train", "test"], default=None, help="Override environment.mode when resuming from models")
    parser.add_argument("--reuse-suffix", action="store_true", help="Use original gs_ suffix from model filename without '-resume' appended")
    # Scheduler options
    parser.add_argument("--procs", type=int, default=0, help="Max parallel jobs (0=auto)")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned jobs, do not launch")
    args = parser.parse_args()

    # Load base config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # SINGLE RUN PATH
    if args.single:
        overrides = {}
        if args.overrides_json:
            try:
                overrides = json.loads(args.overrides_json)
            except Exception as e:
                print(f"Failed to parse --overrides-json: {e}")
                return

        cfg = copy.deepcopy(config)
        for k, v in overrides.items():
            path = k if "." in k else f"environment.{k}"
            _set_by_dot_path(cfg, path, v)

        suffix = args.suffix or ("gs_single" if not overrides else _make_suffix({(k if "." in k else f"environment.{k}"): v for k, v in overrides.items()}))
        cfg = _prepare_logging_dirs(cfg, suffix)

        _run_single(cfg)
        return

    # DEFAULT: simple single run if not grid or grid-from-models
    if not args.grid and not args.grid_from_models:
        _run_single(config)
        return

    # RESUME GRID FROM PRETRAINED MODELS
    if args.grid_from_models:
        # Collect model paths
        model_paths: list[str] = []
        if args.models:
            model_paths.extend([p for p in args.models if p])
        if args.models_file and os.path.exists(args.models_file):
            try:
                with open(args.models_file, "r") as mf:
                    for line in mf:
                        line = line.strip()
                        if line:
                            model_paths.append(line)
            except Exception as e:
                print(f"Failed to read --models-file: {e}")
        if not model_paths:
            print("No model paths provided via --models or --models-file")
            return

        # Build mapping from short key -> full dot-path using current config
        short_to_path: dict[str, str] = {}
        for path, _ in _iter_leaf_paths(config):
            last = path.split(".")[-1]
            short = _short_name(last)
            # prefer environment.* over others if collision
            if short in short_to_path:
                if short_to_path[short].startswith("environment."):
                    continue
            short_to_path[short] = path

        planned = []
        for mp in model_paths:
            base = os.path.basename(mp)
            short_map = _parse_suffix(base)
            if not short_map:
                print(f"Warning: could not parse gs_ suffix from model path: {mp}")
                short_map = {}

            # Translate short keys to full paths
            overrides: dict[str, object] = {}
            for sk, val in short_map.items():
                full = short_to_path.get(sk)
                if full:
                    overrides[full] = val

            # Always set the pretrained model path
            overrides["model.pretrained_model"] = mp

            # Optionally override mode
            if args.resume_mode:
                overrides["environment.mode"] = args.resume_mode

            # Build suffix: reuse parsed or append -resume
            parsed_suffix = None
            if short_map:
                parts = [f"{k}-{short_map[k]}" for k in short_map]
                parsed_suffix = "gs_" + "_".join(parts)
            suffix = parsed_suffix if args.reuse_suffix and parsed_suffix else (parsed_suffix + "-resume" if parsed_suffix else "resume")

            # Convert to bare overrides for environment.*
            bare_overrides: dict[str, object] = {}
            for k, v in overrides.items():
                if k.startswith("environment."):
                    bare_overrides[k.split(".")[-1]] = v
                else:
                    bare_overrides[k] = v

            planned.append((bare_overrides, suffix))

        print(f"Planned resume jobs from {len(model_paths)} pretrained model(s):")
        for i, (bo, suf) in enumerate(planned, 1):
            print(f"  [{i:02d}] suffix={suf} overrides={bo}")

        if args.dry_run:
            print("Dry run: not launching jobs.")
            return

        max_jobs = args.procs if args.procs and args.procs > 0 else max(1, min(len(planned), os.cpu_count()-1 or 1))
        print(f"Launching {len(planned)} jobs with up to {max_jobs} concurrent threads...")

        exe = sys.executable
        script = os.path.abspath(__file__)

        def launch_from_models(bare_overrides: dict, suffix: str):
            cmd = [
                exe,
                script,
                "--single",
                "--config", args.config,
                "--overrides-json", json.dumps(bare_overrides),
                "--suffix", suffix,
            ]
            return subprocess.run(cmd, check=False)

        with futures.ThreadPoolExecutor(max_workers=max_jobs) as pool:
            futs = [pool.submit(launch_from_models, bo, suf) for bo, suf in planned]
            failures = 0
            for i, fut in enumerate(futs, 1):
                res = fut.result()
                if res.returncode != 0:
                    failures += 1
                    print(f"Job #{i} failed with return code {res.returncode}")
            if failures:
                print(f"{failures} job(s) failed.")
            else:
                print("All jobs completed.")
        return

    # GRID SEARCH SCHEDULER PATH (spawns subprocesses controlled by threads)
    explicit_grid = _normalize_grid_keys(config.get("grid_search") or {})
    if not explicit_grid:
        print("grid_search dictionary not found or empty; running single job.")
        _run_single(config)
        return

    value_map: dict[str, list] = dict(explicit_grid)

    # Build Cartesian product of combinations
    param_names = list(value_map.keys())
    grids = [value_map[k] for k in param_names]
    combos = list(itertools.product(*grids))

    print(f"Grid search over {len(param_names)} params with {len(combos)} combinations:")
    planned = []
    for i, values in enumerate(combos, 1):
        combo = dict(zip(param_names, values))
        suffix = _make_suffix(combo)
        # Build overrides JSON with possibly bare names simplified for convenience
        bare_overrides = {k.split(".")[-1]: v for k, v in combo.items() if k.startswith("environment.")}
        # For non-environment keys, keep full path
        for k, v in combo.items():
            if not k.startswith("environment."):
                bare_overrides[k] = v
        planned.append((combo, bare_overrides, suffix))
        print(f"  [{i:02d}] {combo} -> {suffix}")

    if args.dry_run:
        print("Dry run: not launching jobs.")
        return

    max_jobs = args.procs if args.procs and args.procs > 0 else max(1, min(len(planned), os.cpu_count()-1 or 1))
    print(f"Launching {len(planned)} jobs with up to {max_jobs} concurrent threads...")

    exe = sys.executable
    script = os.path.abspath(__file__)

    def launch(bare_overrides: dict, suffix: str):
        cmd = [
            exe,
            script,
            "--single",
            "--config", args.config,
            "--overrides-json", json.dumps(bare_overrides),
            "--suffix", suffix,
        ]
        # Use subprocess to isolate each heavy job (SB3, gym, wandb)
        return subprocess.run(cmd, check=False)

    with futures.ThreadPoolExecutor(max_workers=max_jobs) as pool:
        futs = [pool.submit(launch, bo, suf) for _, bo, suf in planned]
        # Wait for completion and surface return codes
        failures = 0
        for i, fut in enumerate(futs, 1):
            res = fut.result()
            if res.returncode != 0:
                failures += 1
                print(f"Job #{i} failed with return code {res.returncode}")
        if failures:
            print(f"{failures} job(s) failed.")
        else:
            print("All jobs completed.")


if __name__ == "__main__":
    main()
