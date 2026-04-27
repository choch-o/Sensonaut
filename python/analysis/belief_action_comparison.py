
#!/usr/bin/env python3
import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from collections import defaultdict

from typing import Optional, Tuple, Literal, List

# Ensure repo root on path for utils imports
_CUR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import shared constants and utilities
from utils.constants import (
    THETA_GRID, R_GRID,
    GRID_W, GRID_H,
    X_MIN, X_MAX, Z_MIN, Z_MAX,
    HEAD_RADIUS, SPEED_OF_SOUND_US, ITD_NOISE_SCALE,
)
from utils.coordinates import wrap_angle, grid_to_unity, unity_to_grid

# Reuse helpers from existing modules
from human_actions_and_beliefs import (
    load_frames,
    load_agent_frames_generic,
    infer_actions,
    compute_audio_log_like,
    compute_visual_log_like,
)
from compare_traj import load_scene_data, draw_episode_like_maps
from plot_beliefs import _overlay_belief_on_world as overlay_belief_on_world

# Backward compatibility aliases
SPEED_OF_SOUND = SPEED_OF_SOUND_US
ITD_NOISE_US = ITD_NOISE_SCALE

def find_target(scene: dict) -> Optional[dict]:
    if not scene or 'vehicles' not in scene:
        return None
    for v in scene['vehicles']:
        if v.get('isTarget') or 'Target' in v.get('name', ''):
            return v
    return None


# grid_to_unity is now imported from utils.coordinates


def ego_rt_for_step(step: dict, target: dict) -> Tuple[float, float]:
    """Return (r, theta) of target in agent egocentric frame for this step.
    r in meters, theta in radians (−pi..pi, 0=forward, left negative due to our grid convention using atan2(dx,-dz)).
    """
    ax = float(step['position']['x']); az = float(step['position']['z'])
    agent_heading = np.deg2rad(float(step['rotation'].get('y', 0.0)))
    tx = float(target['position']['x']); tz = float(target['position']['z'])
    agx, agy = unity_to_grid(ax, az)
    tgx, tgy = unity_to_grid(tx, tz)
    dx = tgx - agx; dy = tgy - agy
    r = float(np.hypot(dx, dy))
    theta = (math.atan2(dx, -dy) - agent_heading + math.pi) % (2 * math.pi) - math.pi
    
    return r, theta

    # theta_abs = np.arctan2(dx, -dz) % (2 * np.pi)
    # theta_rel = wrap_angle(theta_abs - ayaw)

def sample_at_rt(log_grid: Optional[np.ndarray], r: float, theta: float) -> Optional[float]:
    if log_grid is None:
        return None
    try:
        ir = int(np.argmin(np.abs(R_GRID - r)))
        ith = int(np.argmin(np.abs(THETA_GRID - theta)))
        if 0 <= ir < log_grid.shape[0] and 0 <= ith < log_grid.shape[1]:
            return float(log_grid[ir, ith])
        return None
    except Exception:
        return None

def _rt_bin_for_grid_cell(step, gx, gy):
    """Map a world grid cell (gx, gy) to nearest (r,theta) bin index (ir, ith)."""
    ux, uz = grid_to_unity(gx, gy)
    tmp = {'position': {'x': ux, 'z': uz}}
    r, th = ego_rt_for_step(step, tmp)
    ir = int(np.argmin(np.abs(R_GRID - r)))
    ith = int(np.argmin(np.abs(THETA_GRID - th)))
    return ir, ith

def _logsumexp(arr):
    m = np.nanmax(arr)
    return m + np.log(np.nansum(np.exp(arr - m)) + 1e-12)


def _audio_log_like_geometric(step: dict, scene: dict, target: Optional[dict]) -> np.ndarray:
    """Very simple geometric audio log-like over (r,theta)."""
    thetas = THETA_GRID
    rs = R_GRID

    itd_mixture = []
    weights = []

    r_obj, theta_obj = ego_rt_for_step(step, target)


    itd = (HEAD_RADIUS / SPEED_OF_SOUND) * (abs(theta_obj) + math.sin(abs(theta_obj)))
    itd *= np.sign(theta_obj)

    w = 1.0 / max(r_obj, 1e-6)
    itd_mixture.append(itd)
    weights.append(w)

    if len(weights) == 0:
        itd_obs = np.random.normal(loc=0.0, scale=ITD_NOISE_US)
    else:
        weights = np.array(weights)
        weights /= (np.sum(weights) + 1e-9)
        itd_obs = float(np.sum(weights * np.array(itd_mixture)))
        itd_obs += np.random.normal(loc=0.0, scale=ITD_NOISE_US)

    itd_pred = []
    for theta in thetas:
        itd_model = (HEAD_RADIUS / SPEED_OF_SOUND) * (abs(theta) + math.sin(abs(theta)))
        itd_model *= np.sign(theta)
        itd_pred.append(itd_model)
    itd_pred = np.array(itd_pred)

    log_audio_like_theta = -(itd_obs - itd_pred) ** 2 / (2 * ITD_NOISE_US ** 2) - np.log(math.sqrt(2 * math.pi) * ITD_NOISE_US)
    log_audio_like_theta -= np.max(log_audio_like_theta)
    
    from scipy.ndimage import gaussian_filter1d
    log_audio_like_theta = gaussian_filter1d(
        log_audio_like_theta,
        sigma=1.0 / (thetas[1] - thetas[0]),
        mode="wrap"
    )
    
    log_audio_like = np.tile(log_audio_like_theta, (len(rs), 1))

    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # overlay_belief_on_world(axs[0], step, log_audio_like, alpha=0.1, blur_sigma=(0.7, 1.5), upsample=3, point_size=6)
    # axs[0].set_title("Audio Log-Likelihood Map")
    # overlay_belief_on_world(axs[1], step, step['log_audio'], alpha=0.1, blur_sigma=(0.7, 1.5), upsample=3, point_size=6)
    # axs[1].set_title("Ground Truth Audio Log-Likelihood Map")
    # plt.savefig(f"analysis/plots/audio_log_likelihood_comparison_step_{step['step']}.png")

    return log_audio_like


class BeliefFilter:
    """Filter that fuses visual/audio likelihoods to produce posterior maps."""

    def __init__(self, scene: dict, *, audio_like_source: str, audio_ep_dir: Optional[str] = None):
        self.scene = scene
        self.target = find_target(scene)
        self._audio_source = audio_like_source
        self._audio_ep_dir = audio_ep_dir

        self._last_visual_like = np.ones((len(R_GRID), len(THETA_GRID))) / (len(R_GRID) * len(THETA_GRID))
        self._last_posterior = np.ones((len(R_GRID), len(THETA_GRID))) / (len(R_GRID) * len(THETA_GRID))
        self._last_heading = None
        self.last_agent_pos = None
        # params (mirror env defaults)
        self.sigma_theta_deg = 0.0
        self.alpha_r = 0.0
        self.sigma_r_min = 0.000001
        self.visual_exclusion_decay = 0.5
        self.alpha = 0.8
        self.visual_audio_ratio = 0.7
        self.fov_angle = np.deg2rad(110.0)

        # Cache objects (vehicles) from scene with grid & unity positions
        self._scene_objects = []
        for v in (scene.get('vehicles') or []):
            try:
                ux, uz = v['position']['x'], v['position']['z']
                gx, gy = unity_to_grid(ux, uz)
                self._scene_objects.append({
                    'name': v.get('name', ''),
                    'pos_grid': (gx, gy),
                    'pos_unity': (ux, uz)
                })
            except Exception:
                continue

    def _objects_in_fov(self, step):
        ax_u = float(step['position']['x'])
        az_u = float(step['position']['z'])
        agx, agy = unity_to_grid(ax_u, az_u)
        agent_pos = np.array([float(agx), float(agy)], dtype=float)
        heading = np.deg2rad(float(step['rotation'].get('y', 0.0)))

        # 1) Build occupancy map: grid cell -> object index
        occ_map = {}
        for obj_idx, obj in enumerate(self._scene_objects):
            cx, cy = obj["pos_grid"]
            cx, cy = int(cx), int(cy)
            for gx, gy in footprint_cells_centered(cx, cy):
                occ_map[(gx, gy)] = obj_idx

        # 2) Ray cast per theta within FOV
        agx_g, agy_g = int(agx), int(agy)

        fov_half = float(self.fov_angle) / 2.0
        n_rays = 10  # sample resolution in angle
        step = 0.5    # radial step in grid units

        hits = []  # (obj_idx, r, theta_rel)

        for k in range(n_rays):
            # Relative angle within FOV
            theta_rel = -fov_half + (2.0 * fov_half) * (k / (n_rays - 1))
            theta_abs = heading + theta_rel

            r = 0.0
            while True:
                r += step
                wx = agx_g + int(round(r * math.sin(theta_abs)))
                wy = agy_g + int(round(-r * math.cos(theta_abs)))

                if not (0 <= wx < GRID_W and 0 <= wy < GRID_H):
                    break  # left the grid

                if (wx, wy) in occ_map:
                    obj_idx = occ_map[(wx, wy)]
                    hits.append((obj_idx, r, theta_rel))
                    break  # stop at first occluder on this ray

        # 3) Aggregate hits per object: keep closest hit as representative
        obj_best = {}  # obj_idx -> (r, theta)
        for obj_idx, r, theta in hits:
            if obj_idx not in obj_best or r < obj_best[obj_idx][0]:
                obj_best[obj_idx] = (r, theta)

        visible_objs = [(idx, r, theta) for idx, (r, theta) in obj_best.items()]
        visible_objs.sort(key=lambda x: x[1])  # sort by distance

        out = []
        for i, r, theta in visible_objs:
            obj = self._scene_objects[i]
            out.append({
                'index': i,
                'r': r,
                'theta': theta,
                'name': obj.get('name', ''),
                'pos_unity': obj['pos_unity'],
            })
        return out

    def _car_footprint_map(self):
        CAR_W, CAR_H = 2, 4
        occ = {}
        for i, v in enumerate(self.scene.get('vehicles', [])):
            gx_f, gy_f = unity_to_grid(v['position']['x'], v['position']['z'])
            cx, cy = int(round(gx_f)), int(round(gy_f))

            x0 = cx - CAR_W // 2
            y0 = cy - CAR_H // 2
            for dx in range(CAR_W):
                for dy in range(CAR_H):
                    gx = x0 + dx; gy = y0 + dy
                    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                        occ[(gx, gy)] = i
            # for xx in [gx - 1, gx]:
            #     for yy in [gy - 2, gy - 1, gy, gy + 1]:
            #         occ[(xx, yy)] = i
        return occ

    def _los_visible_free_mask(self, agent_pos_grid):
        thetas, rs = THETA_GRID, R_GRID
        gx, gy = int(agent_pos_grid[0]), int(agent_pos_grid[1])
        mask = np.zeros((len(rs), len(thetas)), dtype=bool)
        if not len(thetas) or not len(rs):
            return mask
        fov_half = float(self.fov_angle) / 2.0
        # thetas are already head-relative in [-pi, pi]; simple abs check
        theta_idx = np.where(np.abs(thetas) <= (fov_half + 1e-9))[0]
        # theta_idx = np.where(np.abs(((thetas + np.pi) % (2 * np.pi) - np.pi)) <= fov_half)[0]
        occ = self._car_footprint_map()
        for j in theta_idx:
            th = float(thetas[j])
            for i, r in enumerate(rs):
                wx = gx + int(r * math.sin(self.agent_heading + th))
                wy = gy + int(-r * math.cos(self.agent_heading + th))
                if not (0 <= wx < GRID_W and 0 <= wy < GRID_H):
                    break
                if (wx, wy) in occ:
                    break
                mask[i, j] = True
        return mask

    def _wrap_angle(self, ang: float) -> float:
        """Wrap angle to [0, 2π)."""
        return ang % (2 * math.pi)

    def _roll_theta(self, arr: np.ndarray, delta_theta: float) -> np.ndarray:
        """Roll a (r,θ) array along θ axis according to delta angle (wrap-around)."""
        if arr is None:
            return arr
        if arr.ndim < 2:
            return arr
        dth = THETA_GRID[1] - THETA_GRID[0]
        shift = int(round(-delta_theta / dth))
        return np.roll(arr, shift, axis=1)


    def update(self, step: dict):
        thetas, rs = THETA_GRID, R_GRID
        ax, az = float(step['position']['x']), float(step['position']['z'])
        heading = np.deg2rad(float(step['rotation'].get('y', 0.0)))
        agx, agy = unity_to_grid(ax, az)
        self.agent_pos = np.array([float(agx), float(agy)], dtype=float)
        self.agent_heading = heading

        # Rotate belief if heading changed
        if self._last_heading is None: 
            self._last_heading = heading
        heading_change = (heading - self._last_heading + math.pi) % (2 * math.pi) - math.pi
        if abs(heading_change) > 1e-6:
            if hasattr(self, "_last_posterior"):
                self._last_posterior = self._roll_theta(self._last_posterior, -heading_change)
            # if hasattr(self, "_last_visual_like"):
            #     self._last_visual_like = self._roll_theta(self._last_visual_like, -heading_change)
            # self._last_heading = heading
            
        # --- Audio likelihood in log-space ---
        if self._audio_source == "geometric":
            log_audio_like = _audio_log_like_geometric(step, self.scene, self.target)
        else:
            log_audio_like = compute_audio_log_like(step, self.scene, self._audio_ep_dir or "")

        if log_audio_like is None:
            log_audio_like = np.zeros((len(R_GRID), len(THETA_GRID)))

        # --- Visual likelihood ---
        if not hasattr(self, "_last_visual_like"):
            self._last_visual_like = np.ones((len(rs), len(thetas))) / (len(rs) * len(thetas))
        visual_like = self._last_visual_like.copy()
        
        # Adjust for head rotation since last step
        heading_change = (heading - self._last_heading + math.pi) % (2 * math.pi) - math.pi
        if abs(heading_change) > 1e-6:
            theta_res = thetas[1] - thetas[0]
            shift_cells = int(round(-heading_change / theta_res))
            visual_like = np.roll(visual_like, shift_cells, axis=1)


        visible = self._objects_in_fov(step)

        sx, sz = grid_cell_sizes_m()

        def _accumulate_visual_peaks(update_map: np.ndarray, cx: int, cy: int, weight: float) -> None:
            for xx, yy in footprint_cells_centered(cx, cy):
                dx_grid = float(xx) - float(self.agent_pos[0])
                dy_grid = float(yy) - float(self.agent_pos[1])
                # r_cell_m = math.hypot(dx_grid * sx, dy_grid * sz)
                r_cell = math.hypot(dx_grid, dy_grid)
                theta_abs = math.atan2(dx_grid, -dy_grid) % (2 * math.pi)
                theta_cell = wrap_angle(theta_abs - self.agent_heading)
                i_r = int(np.argmin(np.abs(rs - r_cell)))
                theta_diffs = (thetas - theta_cell + np.pi) % (2 * np.pi) - np.pi
                i_th = int(np.argmin(np.abs(theta_diffs)))
                # if 0 <= i_r < update_map.shape[0] and 0 <= i_th < update_map.shape[1]:
                update_map[i_r, i_th] += float(weight)

        target_color = None
            
        if self.target is not None:
            name = self.target.get('name', '')
            target_color = 0 if 'Black' in name else (2 if 'White' in name else 1)
        else:
            print("[warn] belief filter: target not found in scene for visual processing")

        update_map = np.zeros_like(visual_like)

        lambda_color = getattr(self, "lambda_color", 0.0)
        self.visible_weight = getattr(self, "visible_weight", 5.0)
        self.visual_alpha = getattr(self, "visual_alpha", 0.7)
    
        for v in visible:
            i_obj = v['index']
            cx = int(round(self._scene_objects[i_obj]['pos_grid'][0]))
            cy = int(round(self._scene_objects[i_obj]['pos_grid'][1]))

            name = self._scene_objects[i_obj].get('name', '')
            obj_color = 0 if 'Black' in name else (2 if 'White' in name else 1)
            w = self.visible_weight if obj_color == target_color else lambda_color
            # print(f"Visible obj: {name} at x={cx}, y={cy}, r={v['r']:.2f}, theta={v['theta']:.2f}, color={obj_color}, weight={w}")
            _accumulate_visual_peaks(update_map, cx, cy, w)

        update_map /= max(1, len(visible))
        visual_like = (1 - self.visual_alpha) * visual_like + self.visual_alpha * update_map
        

        # --- Bayesian Filtering with Log-Space Prior Integration ---
        prior = getattr(self, "_last_posterior", np.ones_like(log_audio_like) / log_audio_like.size)

        # --- Rotate prior to current head-relative frame ---
        if hasattr(self, "_last_heading"):
            heading_change = (heading - self._last_heading + math.pi) % (2 * math.pi) - math.pi
            if abs(heading_change) > 1e-6:
                theta_res = thetas[1] - thetas[0]
                shift_cells = int(round(-heading_change / theta_res))
                prior = np.roll(prior, shift_cells, axis=1)

        # Motion prediction
        if self.last_agent_pos is None:
            self.last_agent_pos = self.agent_pos.copy()
        if hasattr(self, "last_agent_pos"): 
            move_vec = self.agent_pos - self.last_agent_pos
            move_dist = np.linalg.norm(move_vec)
            if move_dist > 0:
                shift_cells = int(round(move_dist / (R_GRID[1] - R_GRID[0])))
                if shift_cells > 0:
                    prior = np.roll(prior, -shift_cells, axis=0)
                    prior[-shift_cells:, :] = np.ones((shift_cells, prior.shape[1])) / (prior.shape[0] * prior.shape[1])



        # --- Visual exclusion/discounting logic ---
        # --- Direct LOS-based visual discounts (applies every step, independent of target visibility) ---
        # 1) Discount LOS-visible FREE-SPACE
        los_free = self._los_visible_free_mask((agx, agy))
        if np.any(los_free):
            visual_like[los_free] *= (1.0 - float(self.visual_exclusion_decay))

        # 2) Discount LOS-visible surfaces of DIFFERENT-COLOR objects (stop at first occluder per theta)
        
        gx, gy = int(agx), int(agy)
        fov_half = float(self.fov_angle) / 2.0
        theta_idx = np.where(np.abs(((thetas + np.pi) % (2 * np.pi) - np.pi)) <= fov_half)[0]
        occ_map = {}
        CAR_W = 2
        CAR_H = 4
        for v in self.scene.get('vehicles', []):
            gx_f, gy_f = unity_to_grid(v['position']['x'], v['position']['z'])
            gx_v, gy_v = int(gx_f), int(gy_f)
            color = 0 if 'Black' in v.get('name', '') else (2 if 'White' in v.get('name', '') else 1)

            x0 = gx_v - CAR_W // 2
            y0 = gy_v - CAR_H // 2
            for dx in range(CAR_W):
                for dy in range(CAR_H):
                    ogx = x0 + dx; ogy = y0 + dy
                    if 0 <= ogx < GRID_W and 0 <= ogy < GRID_H:
                        occ_map[(ogx, ogy)] = color
            # for xx in [gx_v - 1, gx_v]:
            #     for yy in [gy_v - 2, gy_v - 1, gy_v, gy_v + 1]:
            #         occ_map[(xx, yy)] = color

        to_discount = np.zeros_like(visual_like, dtype=bool)
        for j in theta_idx:
            theta_rel = float(thetas[j])
            # March outwards until first occluder
            for i, r in enumerate(rs):
                wx = gx + int(round(r * np.sin(self.agent_heading + theta_rel)))
                wy = gy + int(round(-r * np.cos(self.agent_heading + theta_rel)))
                if not (0 <= wx < GRID_W and 0 <= wy < GRID_H):
                    break
                if (wx, wy) in occ_map:
                    obj_color = occ_map[(wx, wy)]
                    if obj_color != target_color:
                        to_discount[i, j] = True

        if np.any(to_discount):
            visual_like[to_discount] *= (1.0 - float(self.visual_exclusion_decay))

        # Renormalize after discounts
        s = float(np.sum(visual_like))
        if not np.isfinite(s) or s <= 0:
            visual_like = np.ones_like(visual_like) / visual_like.size
        else:
            visual_like /= s

        log_prior = np.log(prior + 1e-12)
        log_visual = np.log(visual_like + 1e-12)
        alpha = self.alpha
        var = self.visual_audio_ratio
        log_post = (1 - alpha) * log_prior + alpha * ((1 - var) * log_audio_like + var * log_visual)
        log_post -= np.max(log_post)

        posterior = np.exp(log_post)
        posterior /= np.sum(posterior) + 1e-9

        self._last_visual_like = visual_like.copy()
        self._last_posterior = posterior.copy()
        self._last_heading = heading
        self.last_agent_pos = np.array([agx, agy], dtype=float)

        return (
            log_visual.copy(),
            log_audio_like.copy(),
            posterior.copy(),
            log_prior.copy(),
        )


def compute_belief_series(
    scene: dict,
    frames: list,
    *,
    audio_like_source: str,
    audio_ep_dir: Optional[str],
) -> Tuple[Tuple[list, list, list, list], Tuple[list, list, list, list]]:
    """Run the belief filter over frames and collect log-map series."""
    filt = BeliefFilter(scene, audio_like_source=audio_like_source, audio_ep_dir=audio_ep_dir)
    target = find_target(scene)

    lv_vals, la_vals, lp_vals, lprior_vals = [], [], [], []
    lv_maps, la_maps, lp_maps, lprior_maps = [], [], [], []
        
    for step in frames:
        try:
            lv, la, lp, lprior = filt.update(step)
            lv_vals.append(footprint_log_at_target(lv, step, target))
            la_vals.append(footprint_log_at_target(la, step, target))
            lp_vals.append(footprint_log_at_target(lp, step, target))
            lprior_vals.append(footprint_log_at_target(lprior, step, target))
            lv_maps.append(lv)
            la_maps.append(la)
            lp_maps.append(lp)
            lprior_maps.append(lprior)
        except Exception as e:
            print(f"[warn] exception in belief series collection: {e}")
            lv_vals.append(None)
            la_vals.append(None)
            lp_vals.append(None)
            lprior_vals.append(None)
            lv_maps.append(None)
            la_maps.append(None)
            lp_maps.append(None)
            lprior_maps.append(None)

    return (lv_vals, la_vals, lp_vals, lprior_vals), (lv_maps, la_maps, lp_maps, lprior_maps)


def collect_saved_belief_series(scene: dict, frames: list, target: Optional[dict]) -> Tuple[Tuple[list, list, list, list], Tuple[list, list, list, list]]:
    """Collect belief series from logs already stored in each frame."""
    lv_vals, la_vals, lp_vals, lprior_vals = [], [], [], []
    lv_maps, la_maps, lp_maps, lprior_maps = [], [], [], []

    for step in frames:
        try:
            lv = step.get('log_visual')
            la = step.get('log_audio')
            lp = step.get('log_posterior')
            lprior = step.get('log_prior')
            lv_vals.append(footprint_log_at_target(lv, step, target))
            la_vals.append(footprint_log_at_target(la, step, target))
            lp_vals.append(footprint_log_at_target(lp, step, target))
            lprior_vals.append(footprint_log_at_target(lprior, step, target) if lprior is not None else None)
            lv_maps.append(lv)
            la_maps.append(la)
            lp_maps.append(lp)
            lprior_maps.append(lprior)
        except Exception as e:
            print(f"[warn] exception in saved belief collection: {e}")
            lv_vals.append(None)
            la_vals.append(None)
            lp_vals.append(None)
            lprior_vals.append(None)
            lv_maps.append(None)
            la_maps.append(None)
            lp_maps.append(None)
            lprior_maps.append(None)
    return (lv_vals, la_vals, lp_vals, lprior_vals), (lv_maps, la_maps, lp_maps, lprior_maps)


def gather_entropy_payloads_for_humans(
    human_root: str,
    audio_like_source: str,
    *,
    map_id: Optional[str] = None,
) -> list:
    payloads = []
    if not os.path.isdir(human_root):
        return payloads
    for pid in sorted(os.listdir(human_root)):
        if not pid.startswith('p'):
            continue
        ep_dir = find_human_episode_dir(human_root, pid, map_id)
        if not ep_dir:
            continue
        scene = load_scene_data(ep_dir)
        if scene is None:
            continue
        frames = load_frames(ep_dir)
        if not frames:
            continue
        scene_map_id = str(scene.get('map_id') or (scene.get('agent') or {}).get('map_id'))
        if map_id is not None and scene_map_id != str(map_id):
            continue
        (_, _, _, _), maps = compute_belief_series(
            scene,
            frames,
            audio_like_source=audio_like_source,
            audio_ep_dir=ep_dir,
        )
        actions, action_times = infer_actions(frames)
        payloads.append({
            "post_maps": maps[2],
            "prior_maps": maps[3],
            "actions": actions,
            "times": action_times,
            "label": pid,
            "map_id": scene_map_id,
        })
    return payloads


def gather_entropy_payloads_for_agents(
    agent_root: str,
    audio_like_source: str,
    agent_belief_mode: str,
    *,
    map_id: Optional[str] = None,
) -> list:
    payloads = []
    if not os.path.isdir(agent_root):
        return payloads
    for agent_id in sorted(os.listdir(agent_root)):
        agent_dir = os.path.join(agent_root, agent_id)
        if not os.path.isdir(agent_dir):
            continue
        trial_dir = find_agent_trial_dir(agent_root, agent_id, map_id)
        if not trial_dir:
            continue
        scene = load_scene_data(trial_dir)
        if scene is None:
            continue
        frames = load_agent_frames_generic(trial_dir)
        if not frames:
            continue
        scene_map_id = str(scene.get('map_id') or (scene.get('agent') or {}).get('map_id'))
        if map_id is not None and scene_map_id != str(map_id):
            continue
        if agent_belief_mode == 'saved':
            target_agent = find_target(scene)
            (_, _, _, _), maps = collect_saved_belief_series(scene, frames, target_agent)
        else:
            (_, _, _, _), maps = compute_belief_series(
                scene,
                frames,
                audio_like_source=audio_like_source,
                audio_ep_dir=trial_dir,
            )
        actions, action_times = infer_actions(frames, agent_mode=True)
        payloads.append({
            "post_maps": maps[2],
            "prior_maps": maps[3],
            "actions": actions,
            "times": action_times,
            "label": agent_id,
            "map_id": scene_map_id,
        })
    return payloads

def _footprint_logmass(log_grid, step, cx, cy):
    """log-sum-exp over the 2x4 footprint bins of the object centered at (cx,cy)."""
    if log_grid is None: return None
    vals = []
    for xx, yy in footprint_cells_centered(cx, cy):
        ir, ith = _rt_bin_for_grid_cell(step, xx, yy)
        if 0 <= ir < log_grid.shape[0] and 0 <= ith < log_grid.shape[1]:
            v = float(log_grid[ir, ith])
            if np.isfinite(v): vals.append(v)
    if not vals: return None
    return _logsumexp(np.array(vals, dtype=float))


def relative_target_metric(log_grid, step, scene, mode="rest"):
    """
    Compute the target's relative advantage in log-space.
      mode in {"rest", "best", "mean"}.
    Returns a float or None.
    """
    target = find_target(scene)
    if target is None or log_grid is None: return None

    # target footprint center (grid coords)
    tgx_f, tgy_f = unity_to_grid(target['position']['x'], target['position']['z'])
    tgx, tgy = int(round(tgx_f)), int(round(tgy_f))
    logmass_T = _footprint_logmass(log_grid, step, tgx, tgy)
    if logmass_T is None: return None

    if mode == "rest":
        # all cells except target footprint
        all_vals = np.asarray(log_grid).ravel()
        # subtract target footprint in log-space: do it by “masking out” indices
        # Build mask by collecting footprint indices
        idxs = []
        for xx, yy in footprint_cells_centered(tgx, tgy):
            ir, ith = _rt_bin_for_grid_cell(step, xx, yy)
            if 0 <= ir < log_grid.shape[0] and 0 <= ith < log_grid.shape[1]:
                idxs.append(ir * log_grid.shape[1] + ith)
        mask = np.ones(all_vals.size, dtype=bool)
        for k in idxs:
            if 0 <= k < mask.size: mask[k] = False
        rest_vals = all_vals[mask]
        if rest_vals.size == 0: return None
        logmass_rest = _logsumexp(rest_vals)
        return logmass_T - logmass_rest

    # object-wise alternatives
    other_logmasses = []
    for v in (scene.get('vehicles') or []):
        if v is target: continue
        gx_f, gy_f = unity_to_grid(v['position']['x'], v['position']['z'])
        gx, gy = int(round(gx_f)), int(round(gy_f))
        lm = _footprint_logmass(log_grid, step, gx, gy)
        if lm is not None: other_logmasses.append(lm)

    if not other_logmasses:
        return None

    if mode == "best":
        return logmass_T - float(np.max(other_logmasses))

    # mode == "mean"
    # log(mean(sumexp footprints)) = logsumexp(other_logmasses) - log(K)
    K = float(len(other_logmasses))
    logsum_others = _logsumexp(np.array(other_logmasses, dtype=float))
    logmean_others = logsum_others - np.log(K + 1e-12)
    return logmass_T - logmean_others

def footprint_log_at_target(log_grid: Optional[np.ndarray], step: dict, target: dict) -> Optional[float]:
    if log_grid is None:
        return None
    try:
        cx_f, cy_f = unity_to_grid(target['position']['x'], target['position']['z'])
        cx_i, cy_i = int(round(cx_f)), int(round(cy_f))
        cells = [(cx_i + dx, cy_i + dy) for dx in (-1, 0) for dy in (-2, -1, 0, 1)]
        vals = []
        for gx, gy in cells:
            ux, uz = grid_to_unity(gx, gy)
            tmp_t = {'position': {'x': ux, 'z': uz}}
            r, th = ego_rt_for_step(step, tmp_t)
            v = sample_at_rt(log_grid, r, th)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        if not vals:
            return None
        m = max(vals)
        return m #+ np.log(np.mean(np.exp(np.array(vals) - m) + 1e-12))
    except Exception:
        return None


def maps_index_for_pid(maps_json: str, pid: str, map_id: str) -> Optional[str]:
    with open(maps_json, 'r', encoding='utf-8') as f:
        md = json.load(f)
    arr = md.get(pid, [])
    for i, m in enumerate(arr):
        if str(m.get('map_id')) == str(map_id) or str(m.get('id')) == str(map_id):
            return str(i)
    return None


def find_human_episode_dir(human_root: str, pid: str, map_id: str) -> Optional[str]:
    maps_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'maps', 'maps.json')
    idx = maps_index_for_pid(maps_json, pid, map_id)
    if idx is None:
        return None
    ep_dir = os.path.join(human_root, pid, idx)
    return ep_dir if os.path.isdir(ep_dir) else None


def find_agent_trial_dir(agent_root: str, agent_id: str, map_id: str) -> Optional[str]:
    base = os.path.join(agent_root, agent_id)
    if not os.path.isdir(base):
        return None
    trials = [d for d in glob.glob(os.path.join(base, '*')) if os.path.isdir(d)]
    for t in sorted(trials):
        scene_files = glob.glob(os.path.join(t, 'scene_data_*.json'))
        if not scene_files:
            continue
        try:
            with open(scene_files[0], 'r', encoding='utf-8') as f:
                sj = json.load(f)
            mid = sj.get('map_id') or (sj.get('agent') or {}).get('map_id')
            if str(mid) == str(map_id):
                return t
        except Exception:
            pass
    return None

def _get_ts(step, default=None):
    """Read a timestamp-like field if present, else default."""
    for k in ('time','timestamp','ts','t','frameTime'):
        v = step.get(k)
        if v is not None:
            try: return float(v)
            except Exception: pass
    for k in ('frame','step','idx','index'):
        v = step.get(k)
        if v is not None:
            try: return float(v)
            except Exception: pass
    return default

def grid_cell_sizes_m() -> Tuple[float, float]:
    """Meters per grid cell in x(z) from the scene extents."""
    gw = max(1, GRID_W - 1)
    gh = max(1, GRID_H - 1)
    sx = (X_MAX - X_MIN) / gw
    sz = (Z_MAX - Z_MIN) / gh  # positive magnitude
    return float(sx), float(abs(sz))

def footprint_cells_centered(cx: int, cy: int):
    """The 2×4 footprint used elsewhere, centered around (cx, cy)."""
    # for xx in (cx - 1, cx):
    #     for yy in (cy - 2, cy - 1, cy, cy + 1):
    #         yield xx, yy
    # xs = [cx - 1, cx]
    # ys = [cy - 2, cy - 1, cy, cy + 1]
    xs = [cx - 1, cx]
    ys = [cy - 2, cy - 1, cy, cy + 1]
    for xx in xs:
        if xx < 0 or xx >= GRID_W:
            continue
        for yy in ys:
            if yy < 0 or yy >= GRID_H:
                continue
            yield xx, yy

def build_agent_to_human_match(h_frames, a_frames):
    """
    For each HUMAN index j, return the index i of the closest AGENT step.
    If both have timestamps, match by time; else fall back to proportional mapping.
    """
    h_ts = [ _get_ts(s, default=None) for s in h_frames ]
    a_ts = [ _get_ts(s, default=None) for s in a_frames ]
    have_times = (all(t is not None for t in h_ts) and all(t is not None for t in a_ts))

    if have_times:
        a_ts_np = np.asarray(a_ts, dtype=float)
        mapping = []
        for tj in h_ts:
            i = int(np.argmin(np.abs(a_ts_np - float(tj))))
            mapping.append(i)
        return mapping

    # proportional fallback: human j -> nearest agent index on normalized progress
    H = max(1, len(h_frames)-1)
    A = max(1, len(a_frames)-1)
    mapping = []
    for j in range(len(h_frames)):
        i = int(round(j * A / H))
        i = max(0, min(len(a_frames)-1, i))
        mapping.append(i)
    return mapping


def build_human_to_agent_match(h_frames, a_frames):
    """
    For each agent step i, return the index j of the closest human step.
    If timestamps exist in both, match by time; otherwise use a length-proportional fallback.
    """
    # try timestamps
    h_ts = [ _get_ts(s, default=None) for s in h_frames ]
    a_ts = [ _get_ts(s, default=None) for s in a_frames ]
    have_times = (all(t is not None for t in h_ts) and all(t is not None for t in a_ts))

    if have_times:
        # nearest neighbor in time
        h_ts_np = np.asarray(h_ts, dtype=float)
        match = []
        for t in a_ts:
            j = int(np.argmin(np.abs(h_ts_np - float(t))))
            match.append(j)
        return match

    # fallback: proportional mapping [0..len-1] → [0..len-1]
    H = max(1, len(h_frames)-1)
    A = max(1, len(a_frames)-1)
    match = []
    for i in range(len(a_frames)):
        # map agent i to a human index j on the same normalized progress
        j = int(round(i * H / A))
        j = max(0, min(len(h_frames)-1, j))
        match.append(j)
    return match


def get_time_axis(frames):
    ts = []
    for s in frames:
        t = _get_ts(s, default=None)
        ts.append(float(t) if t is not None else None)
    # fallback to 0..N-1 if no usable times
    if any(t is None for t in ts):
        return np.arange(len(frames), dtype=float)
    return np.asarray(ts, dtype=float)

def _scatter_actions_at(ax, actions, x_positions, color, offset=0.0):
    ymap = {'stay':0,'turn_left':1,'turn_right':1.3,'step_forward':2,'commit':2.6}
    ys = [ ymap.get(a, 0) + offset for a in actions ]
    ax.scatter(x_positions, ys, s=28, color=color, alpha=0.75)

def remap_by_index(src_series, index_list, N_out):
    """Take src_series[index_list[k]] for k in 0..N_out-1."""
    out = []
    for k in range(N_out):
        i = index_list[k]
        if i is None or i < 0 or i >= len(src_series):
            out.append(None)
        else:
            out.append(src_series[i])
    return out

def footprint_prob_at_target(log_grid, step, target):
    if log_grid is None:
        return None
    # compute log-mean-exp over 2x4 footprint cells at the target
    lg = footprint_log_at_target(log_grid, step, target)
    if lg is None:
        return None
    # logsumexp over full grid (for numerical stability)
    m = np.nanmax(log_grid)
    lse = m + np.log(np.nansum(np.exp(log_grid - m)) + 1e-12)
    # p(target) = exp(log_target - logsumexp)
    return float(np.exp(lg - lse))

MAX_BELIEF_ENTROPY = math.log(max(1, len(R_GRID) * len(THETA_GRID)))


def _scene_map_id(scene: Optional[dict]) -> Optional[str]:
    if not scene:
        return None
    mid = scene.get('map_id')
    if mid is None:
        agent_info = scene.get('agent')
        if isinstance(agent_info, dict):
            mid = agent_info.get('map_id')
    return str(mid) if mid is not None else None


def total_distance_moved(frames: List[dict]) -> Optional[float]:
    coords: List[Tuple[float, float]] = []
    for step in frames or []:
        pos = step.get('position') if isinstance(step, dict) else None
        if not isinstance(pos, dict):
            continue
        x = pos.get('x')
        z = pos.get('z')
        if x is None or z is None:
            continue
        try:
            coords.append((float(x), float(z)))
        except Exception:
            continue
    if len(coords) < 2:
        return None
    dist = 0.0
    for (x0, z0), (x1, z1) in zip(coords[:-1], coords[1:]):
        dist += math.hypot(x1 - x0, z1 - z0)
    return float(dist)


def _gather_human_distances_by_map(human_root: str) -> tuple[list[float], dict]:
    overall: list[float] = []
    by_map: dict[str, list[float]] = defaultdict(list)
    if not os.path.isdir(human_root):
        return overall, by_map
    for pid in sorted(os.listdir(human_root)):
        if not pid.startswith('p'):
            continue
        pid_dir = os.path.join(human_root, pid)
        if not os.path.isdir(pid_dir):
            continue
        for sub in sorted(os.listdir(pid_dir)):
            ep_dir = os.path.join(pid_dir, sub)
            if not os.path.isdir(ep_dir):
                continue
            scene = load_scene_data(ep_dir)
            map_id = _scene_map_id(scene)
            frames = load_frames(ep_dir)
            dist = total_distance_moved(frames)
            if dist is None:
                continue
            overall.append(dist)
            bucket = str(map_id) if map_id is not None else "unknown"
            by_map[bucket].append(dist)
    return overall, by_map


def gather_total_distances_for_humans(human_root: str, *, map_id: Optional[str] = None) -> List[float]:
    distances: List[float] = []
    if not os.path.isdir(human_root):
        return distances
    target_map = str(map_id) if map_id is not None else None
    for pid in sorted(os.listdir(human_root)):
        if not pid.startswith('p'):
            continue
        pid_dir = os.path.join(human_root, pid)
        if not os.path.isdir(pid_dir):
            continue
        for sub in sorted(os.listdir(pid_dir)):
            ep_dir = os.path.join(pid_dir, sub)
            if not os.path.isdir(ep_dir):
                continue
            if target_map is not None:
                scene = load_scene_data(ep_dir)
                if _scene_map_id(scene) != target_map:
                    continue
            frames = load_frames(ep_dir)
            dist = total_distance_moved(frames)
            if dist is None:
                continue
            distances.append(dist)
    return distances


def _gather_agent_distances_by_map(agent_root: str, *, agent_ids: Optional[List[str]] = None) -> tuple[list[float], dict]:
    overall: list[float] = []
    by_map: dict[str, list[float]] = defaultdict(list)
    if not os.path.isdir(agent_root):
        return overall, by_map
    if agent_ids is None:
        candidates = sorted(
            d for d in os.listdir(agent_root)
            if os.path.isdir(os.path.join(agent_root, d))
        )
    else:
        candidates = agent_ids
    for agent_id in candidates:
        base = os.path.join(agent_root, agent_id)
        if not os.path.isdir(base):
            continue
        trials = sorted(
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        )
        for trial in trials:
            trial_dir = os.path.join(base, trial)
            scene = load_scene_data(trial_dir)
            map_id = _scene_map_id(scene)
            frames = load_agent_frames_generic(trial_dir)
            dist = total_distance_moved(frames)
            if dist is None:
                continue
            overall.append(dist)
            bucket = str(map_id) if map_id is not None else "unknown"
            by_map[bucket].append(dist)
    return overall, by_map


def gather_total_distances_for_agents(
    agent_root: str,
    *,
    agent_ids: Optional[List[str]] = None,
    map_id: Optional[str] = None,
) -> List[float]:
    distances: List[float] = []
    if not os.path.isdir(agent_root):
        return distances
    if agent_ids is None:
        candidates = sorted(
            d for d in os.listdir(agent_root)
            if os.path.isdir(os.path.join(agent_root, d))
        )
    else:
        candidates = agent_ids
    target_map = str(map_id) if map_id is not None else None
    for agent_id in candidates:
        base = os.path.join(agent_root, agent_id)
        if not os.path.isdir(base):
            continue
        trials = sorted(
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        )
        for trial in trials:
            trial_dir = os.path.join(base, trial)
            if target_map is not None:
                scene = load_scene_data(trial_dir)
                if _scene_map_id(scene) != target_map:
                    continue
            frames = load_agent_frames_generic(trial_dir)
            dist = total_distance_moved(frames)
            if dist is None:
                continue
            distances.append(dist)
    return distances


def gather_total_distances_by_map(
    human_root: str,
    agent_root: str,
    *,
    agent_ids: Optional[List[str]] = None,
) -> tuple[list[float], list[float], dict, dict]:
    """Return (human_all, agent_all, human_by_map, agent_by_map)."""
    human_all, human_by_map = _gather_human_distances_by_map(human_root)
    agent_all, agent_by_map = _gather_agent_distances_by_map(agent_root, agent_ids=agent_ids)
    return human_all, agent_all, human_by_map, agent_by_map


def plot_distance_histogram(
    out_path: str,
    *,
    human_root: str,
    agent_root: str,
    map_id: Optional[str] = None,
    agent_ids: Optional[List[str]] = None,
    human_dists: Optional[List[float]] = None,
    agent_dists: Optional[List[float]] = None,
    bins: int = 20,
) -> None:
    if human_dists is None:
        human_dists = gather_total_distances_for_humans(human_root, map_id=map_id)
    if agent_dists is None:
        agent_dists = gather_total_distances_for_agents(agent_root, agent_ids=agent_ids, map_id=map_id)

    print(f"[info] collected {len(human_dists)} human distances and {len(agent_dists)} agent distances")
    if not human_dists and not agent_dists:
        print(f"[warn] distance histogram skipped: no distance data")
        return

    all_vals = human_dists + agent_dists
    max_val = 10.0
    bin_edges = np.linspace(0.0, max_val * 1.05, bins + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    if human_dists:
        ax.hist(
            human_dists,
            bins=bin_edges,
            color='#e07a7a',
            alpha=0.6,
            edgecolor='white',
            label=f"Humans (n={len(human_dists)})",
        )
        ax.set_xticks(np.linspace(0,10,21))
        ax.set_xlim(0, max_val * 1.05)
        ax.axvline(np.median(human_dists), color='#e07a7a', lw=1.1, ls='--', alpha=0.8)
        
    if agent_dists:
        ax.hist(
            agent_dists,
            bins=bin_edges,
            color='#1b9e77',
            alpha=0.6,
            edgecolor='white',
            label=f"Agents (n={len(agent_dists)})",
        )
        ax.set_xticks(np.linspace(0,10,21))
        ax.set_xlim(0, max_val * 1.05)
        ax.axvline(np.median(agent_dists), color='#1b9e77', lw=1.1, ls='--', alpha=0.8)
        

    ax.set_xlabel('Distance moved (Unity units)')
    ax.set_ylabel('Episode count')
    title = 'Distance moved distribution'
    if map_id is not None:
        title += f' — map {map_id}'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'[ok] saved {out_path}')


def _condition_label(cond: Optional[dict]) -> str:
    if not cond:
        return "unknown"
    parts = [f"{k}={v}" for k, v in sorted(cond.items())]
    return ", ".join(parts)


def plot_distance_histogram_by_map(
    out_path: str,
    *,
    map_ids: List[str],
    human_by_map: dict,
    agent_by_map: dict,
    bins: int = 20,
    max_val: float = 10.0,
) -> None:
    if not map_ids:
        print("[warn] per-map distance histogram skipped: no map ids provided")
        return

    bin_edges = np.linspace(0.0, max_val * 1.05, bins + 1)
    n_maps = len(map_ids)
    ncols = min(15, max(1, int(math.ceil(math.sqrt(n_maps)))))
    nrows = int(math.ceil(n_maps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    human_color = '#e07a7a'
    agent_color = '#1b9e77'
    human_handle = None
    agent_handle = None

    for idx, mid in enumerate(map_ids):
        ax = axes[idx]
        h_vals = human_by_map.get(mid, [])
        a_vals = agent_by_map.get(mid, [])
        has_any = False
        if h_vals:
            h_hist = ax.hist(
                h_vals, bins=bin_edges, color=human_color, alpha=0.6, edgecolor='white'
            )
            ax.axvline(np.median(h_vals), color=human_color, lw=0.8, ls='--', alpha=0.8)
            has_any = True
            if human_handle is None:
                human_handle = h_hist[2][0]
        if a_vals:
            a_hist = ax.hist(
                a_vals, bins=bin_edges, color=agent_color, alpha=0.6, edgecolor='white'
            )
            ax.axvline(np.median(a_vals), color=agent_color, lw=0.8, ls='--', alpha=0.8)
            has_any = True
            if agent_handle is None:
                agent_handle = a_hist[2][0]
        if not has_any:
            ax.set_facecolor('#f9f9f9')
            ax.text(0.5, 0.5, "no data", ha='center', va='center', fontsize=6, transform=ax.transAxes, color='#666666')
        ax.set_title(mid, fontsize=7)
        ax.set_xlim(0, max_val * 1.05)
        ax.set_xticks([0, 5, 10])
        ax.tick_params(labelsize=6)
        ax.text(
            0.98, 0.90, f"H:{len(h_vals)} A:{len(a_vals)}",
            ha='right', va='top', fontsize=6, transform=ax.transAxes, color='#333333'
        )

    for ax in axes[n_maps:]:
        ax.axis('off')

    fig.supxlabel('Distance moved (Unity units)')
    fig.supylabel('Episode count')
    fig.suptitle('Distance moved per map', fontsize=12)
    handles = []
    labels = []
    if human_handle is not None:
        handles.append(human_handle)
        labels.append('Humans')
    if agent_handle is not None:
        handles.append(agent_handle)
        labels.append('Agents')
    if handles:
        fig.legend(
            handles,
            labels,
            loc='upper right',
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.98, 0.995)
        )
    plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'[ok] saved {out_path}')


def plot_distance_histogram_by_condition(
    out_path: str,
    *,
    map_ids: List[str],
    map_conditions: dict,
    human_by_map: dict,
    agent_by_map: dict,
    bins: int = 20,
    max_val: float = 10.0,
) -> None:
    if not map_ids:
        print("[warn] per-condition distance histogram skipped: no map ids provided")
        return

    human_by_cond: dict[str, list[float]] = defaultdict(list)
    agent_by_cond: dict[str, list[float]] = defaultdict(list)
    for mid in map_ids:
        label = _condition_label(map_conditions.get(mid))
        if human_by_map.get(mid):
            human_by_cond[label].extend(human_by_map[mid])
        if agent_by_map.get(mid):
            agent_by_cond[label].extend(agent_by_map[mid])

    cond_labels = sorted(set(human_by_cond.keys()) | set(agent_by_cond.keys()))
    if not cond_labels:
        print("[warn] per-condition distance histogram skipped: no data")
        return

    bin_edges = np.linspace(0.0, max_val * 1.05, bins + 1)
    n_conds = len(cond_labels)
    ncols = min(4, max(1, n_conds))
    nrows = int(math.ceil(n_conds / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    human_color = '#e07a7a'
    agent_color = '#1b9e77'
    human_handle = None
    agent_handle = None

    for idx, label in enumerate(cond_labels):
        ax = axes[idx]
        h_vals = human_by_cond.get(label, [])
        a_vals = agent_by_cond.get(label, [])
        has_any = False
        if h_vals:
            h_hist = ax.hist(h_vals, bins=bin_edges, color=human_color, alpha=0.6, edgecolor='white')
            ax.axvline(np.median(h_vals), color=human_color, lw=1.0, ls='--', alpha=0.8)
            has_any = True
            if human_handle is None:
                human_handle = h_hist[2][0]
        if a_vals:
            a_hist = ax.hist(a_vals, bins=bin_edges, color=agent_color, alpha=0.6, edgecolor='white')
            ax.axvline(np.median(a_vals), color=agent_color, lw=1.0, ls='--', alpha=0.8)
            has_any = True
            if agent_handle is None:
                agent_handle = a_hist[2][0]
        if not has_any:
            ax.set_facecolor('#f9f9f9')
            ax.text(0.5, 0.5, "no data", ha='center', va='center', fontsize=8, transform=ax.transAxes, color='#666666')
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, max_val * 1.05)
        ax.set_xticks([0, 5, 10])
        ax.tick_params(labelsize=8)
        ax.text(
            0.98, 0.92, f"H:{len(h_vals)} A:{len(a_vals)}",
            ha='right', va='top', fontsize=7, transform=ax.transAxes, color='#333333'
        )

    for ax in axes[n_conds:]:
        ax.axis('off')

    fig.supxlabel('Distance moved (Unity units)')
    fig.supylabel('Episode count')
    fig.suptitle('Distance moved by condition (maps grouped)', fontsize=14)
    handles = []
    labels = []
    if human_handle is not None:
        handles.append(human_handle)
        labels.append('Humans')
    if agent_handle is not None:
        handles.append(agent_handle)
        labels.append('Agents')
    if handles:
        fig.legend(
            handles,
            labels,
            loc='upper right',
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.98, 0.995)
        )
    plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f'[ok] saved {out_path}')


def _entropy_from_log_grid(log_grid: Optional[np.ndarray]) -> float:
    """Return (natural-log) entropy of a belief stored in log-space."""
    if log_grid is None:
        return float('nan')
    arr = np.asarray(log_grid, dtype=float).ravel()
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float('nan')
    arr = arr[mask]
    logZ = _logsumexp(arr)
    probs = np.exp(arr - logZ)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.sum(probs * (arr - logZ))
    return float(entropy)


def plot_entropy_eig_relationships(
    out_path: str,
    *,
    human: dict,
    agent: dict,
    explore_actions: Optional[set] = None,
    num_bins: int = 12,
) -> None:
    """Plot entropy/EIG trade-offs versus action selection for human and agent."""
    explore_actions = explore_actions or {'turn_left', 'turn_right', 'step_forward'}

    def _prepare_series(data: dict) -> dict:
        entries = data.get('series')
        if entries is None:
            entries = [data]
        label = data.get('label', 'dataset')

        entropy_parts = []
        info_gain_parts = []
        eig_parts = []
        explore_parts = []
        non_commit_parts = []

        for entry in entries:
            post_maps = entry.get('post_maps') or []
            prior_maps = entry.get('prior_maps') or []
            actions = entry.get('actions') or []
            times = entry.get('times') or []
            n = min(len(post_maps), len(prior_maps), len(actions), len(times))
            if n == 0:
                continue
            entropy = np.full(n, np.nan)
            info_gain = np.full(n, np.nan)
            for i in range(n):
                lp = post_maps[i]
                pr = prior_maps[i] if i < len(prior_maps) else None
                if lp is None or pr is None:
                    continue
                post_h = _entropy_from_log_grid(lp)
                prior_h = _entropy_from_log_grid(pr)
                if not np.isfinite(post_h) or not np.isfinite(prior_h):
                    continue
                entropy[i] = post_h
                info_gain[i] = prior_h - post_h

            times_arr = np.asarray(times[:n], dtype=float) if n else np.empty(0, dtype=float)
            dt = np.full(n, np.nan)
            if times_arr.size == n and n > 1:
                dt[:-1] = np.diff(times_arr)
                dt[dt <= 0] = np.nan

            with np.errstate(divide='ignore', invalid='ignore'):
                eig_per_cost = np.divide(info_gain, dt, out=np.full_like(info_gain, np.nan), where=dt > 0)

            if MAX_BELIEF_ENTROPY > 0:
                entropy_norm = entropy / MAX_BELIEF_ENTROPY
                info_gain_norm = info_gain / MAX_BELIEF_ENTROPY
                eig_per_cost_norm = np.divide(info_gain_norm, dt, out=np.full_like(info_gain_norm, np.nan), where=dt > 0)
            else:
                entropy_norm = entropy
                info_gain_norm = info_gain
                eig_per_cost_norm = eig_per_cost

            explore_flag = np.zeros(n, dtype=float)
            non_commit = np.zeros(n, dtype=bool)
            for i in range(n):
                if i >= len(actions):
                    continue
                act = actions[i]
                explore_flag[i] = 1.0 if act in explore_actions else 0.0
                non_commit[i] = (act != 'commit')

            entropy_parts.append(entropy_norm)
            info_gain_parts.append(info_gain_norm)
            eig_parts.append(eig_per_cost_norm)
            explore_parts.append(explore_flag)
            non_commit_parts.append(non_commit)

        def _concat(parts, dtype=float):
            parts = [p for p in parts if isinstance(p, np.ndarray) and p.size]
            if not parts:
                return np.empty(0, dtype=dtype)
            return np.concatenate(parts, axis=0).astype(dtype, copy=False)

        return {
            'label': label,
            'entropy': _concat(entropy_parts, dtype=float),
            'info_gain': _concat(info_gain_parts, dtype=float),
            'eig_per_cost': _concat(eig_parts, dtype=float),
            'explore_flag': _concat(explore_parts, dtype=float),
            'non_commit': _concat(non_commit_parts, dtype=bool),
        }

    human_series = _prepare_series(human)
    agent_series = _prepare_series(agent)

    datasets = [
        (human_series, '#e07a7a'),
        (agent_series, '#1b9e77'),
    ]

    def _binned_curve(x_vals: np.ndarray, y_vals: np.ndarray, bins: np.ndarray):
        if x_vals.size == 0:
            return np.array([]), np.array([]), np.array([])
        idx = np.digitize(x_vals, bins) - 1
        centers, probs, counts = [], [], []
        for i in range(len(bins) - 1):
            mask = idx == i
            if not np.any(mask):
                continue
            centers.append(0.5 * (bins[i] + bins[i + 1]))
            probs.append(float(np.mean(y_vals[mask])))
            counts.append(int(np.sum(mask)))
        return np.asarray(centers), np.asarray(probs), np.asarray(counts)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: P(explore) vs entropy ---
    bins_entropy = np.linspace(0.0, 1.0, num_bins + 1)
    plotted_entropy = False
    for series, color in datasets:
        mask = np.isfinite(series['entropy']) & np.isfinite(series['explore_flag']) & series['non_commit']
        x = series['entropy'][mask]
        y = series['explore_flag'][mask]
        if x.size == 0:
            continue
        centers, probs, _ = _binned_curve(x, y, bins_entropy)
        if centers.size == 0:
            continue
        axes[0].plot(centers, probs, color=color, lw=1.8, marker='o', label=series['label'])
        plotted_entropy = True
    axes[0].axhline(0.5, color='k', lw=1.0, ls='--', alpha=0.35)
    axes[0].set_xlabel('Belief entropy H(b) (normalized)')
    axes[0].set_ylabel('P(explore action)')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    if plotted_entropy:
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[0].transAxes, fontsize=9)

    # --- Plot 2: P(explore) vs EIG per cost ---
    eig_values = []
    for series, _ in datasets:
        mask = np.isfinite(series['eig_per_cost']) & series['non_commit']
        eig_values.append(series['eig_per_cost'][mask])
    eig_values = np.concatenate([v for v in eig_values if v.size], axis=0) if any(v.size for v in eig_values) else np.array([])
    if eig_values.size:
        eig_min = float(np.nanmin(eig_values))
        eig_max = float(np.nanmax(eig_values))
        if not np.isfinite(eig_min) or not np.isfinite(eig_max):
            eig_values = np.array([])
    if eig_values.size:
        if eig_min == eig_max:
            eig_min -= 0.1
            eig_max += 0.1
        bins_eig = np.linspace(eig_min, eig_max, num_bins + 1)
    else:
        bins_eig = None

    plotted_eig = False
    if bins_eig is not None:
        for series, color in datasets:
            mask = np.isfinite(series['eig_per_cost']) & np.isfinite(series['explore_flag']) & series['non_commit']
            x = series['eig_per_cost'][mask]
            y = series['explore_flag'][mask]
            if x.size == 0:
                continue
            centers, probs, _ = _binned_curve(x, y, bins_eig)
            if centers.size == 0:
                continue
            axes[1].plot(centers, probs, color=color, lw=1.8, marker='o', label=series['label'])
            plotted_eig = True
        axes[1].axhline(0.5, color='k', lw=1.0, ls='--', alpha=0.35)
        axes[1].set_xlabel('EIG per cost (normalized)')
        axes[1].set_ylabel('P(explore action)')
        axes[1].grid(True, alpha=0.3)
        if plotted_eig:
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[1].transAxes, fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[1].transAxes, fontsize=9)
        axes[1].set_xlabel('EIG per cost (normalized)')
        axes[1].set_ylabel('P(explore action)')
        axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Entropy vs EIG per cost scatter ---
    plotted_scatter = False
    for series, color in datasets:
        mask = np.isfinite(series['entropy']) & np.isfinite(series['eig_per_cost']) & series['non_commit']
        x = series['entropy'][mask]
        y = series['eig_per_cost'][mask]
        if x.size == 0:
            continue
        axes[2].scatter(x, y, color=color, alpha=0.35, s=28, label=series['label'])
        plotted_scatter = True
    axes[2].set_xlabel('Belief entropy H(b) (normalized)')
    axes[2].set_ylabel('EIG per cost (normalized)')
    axes[2].grid(True, alpha=0.3)
    if plotted_scatter:
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[2].transAxes, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f'[ok] saved {out_path}')

def _reserve_space_for_minimaps(*, rows, row_h=0.18, vgap=0.012, bottom=0.05,
                                left=0.08, right=0.98, top=0.96, hspace=0.4):
    """Lift the main subplots so the minimap block (placed at `bottom`) won't overlap."""
    block_h = rows * row_h + (rows - 1) * vgap
    main_bottom = bottom + block_h + 0.02   # + a little safety margin
    plt.subplots_adjust(left=left, right=right, top=top, bottom=main_bottom, hspace=hspace)
    return main_bottom  # if you want to reuse it

def plot_belief_action(
    h_ep: str,
    a_ep: str,
    out_path: str,
    audio_like_source: str = "wav",
    agent_belief_mode: str = "recompute",
):
    h_scene = load_scene_data(h_ep)
    a_scene = load_scene_data(a_ep)
    scene = h_scene or a_scene
    if scene is None:
        print('[error] no scene found')
        return

    human_scene = h_scene or scene
    agent_scene = a_scene or scene

    h_frames = load_frames(h_ep)
    a_frames = load_agent_frames_generic(a_ep)
    if not h_frames or not a_frames:
        print('[error] missing frames')
        return

    print("[info] computing HUMAN belief series...")
    (h_lv, h_la, h_lp, h_lprior), (h_lv_maps, h_la_maps, h_lp_maps, h_lprior_maps) = compute_belief_series(
        human_scene,
        h_frames,
        audio_like_source=audio_like_source,
        audio_ep_dir=h_ep,
    )

    if agent_belief_mode == 'saved':
        target_agent = find_target(agent_scene)
        (a_lv, a_la, a_lp, a_lprior), (a_lv_maps, a_la_maps, a_lp_maps, a_lprior_maps) = collect_saved_belief_series(
            a_scene,
            a_frames,
            target_agent,
        )
    else:
        print("[info] computing AGENT belief series...")
        audio_ref_dir = h_ep if audio_like_source != 'geometric' else a_ep
        (a_lv, a_la, a_lp, a_lprior), (a_lv_maps, a_la_maps, a_lp_maps, a_lprior_maps) = compute_belief_series(
            agent_scene,
            a_frames,
            audio_like_source=audio_like_source,
            audio_ep_dir=audio_ref_dir,
        )

    

    # --- NEW: relative target-advantage series for each map type ---
    rel_modes = ["best", "mean", "rest"]
    kinds = {
        "visual":    (h_lv_maps, a_lv_maps),
        "audio":     (h_la_maps, a_la_maps),
        "posterior": (h_lp_maps, a_lp_maps),
    }

    # h_rel[kind][mode] = list over human steps
    # a_rel[kind][mode] = list over agent steps
    h_rel = {k: {m: [] for m in rel_modes} for k in kinds}
    a_rel = {k: {m: [] for m in rel_modes} for k in kinds}

    # Human-relative series
    for j, h_step in enumerate(h_frames):
        for kind, (Hmaps, Amaps) in kinds.items():
            grid = Hmaps[j] if j < len(Hmaps) else None
            for m in rel_modes:
                h_rel[kind][m].append(relative_target_metric(grid, h_step, scene, mode=m))

    # Agent-relative series
    for i, a_step in enumerate(a_frames):
        for kind, (Hmaps, Amaps) in kinds.items():
            grid = Amaps[i] if i < len(Amaps) else None
            for m in rel_modes:
                a_rel[kind][m].append(relative_target_metric(grid, a_step, scene, mode=m))

    # index maps (both directions)
    human_to_agent = build_agent_to_human_match(h_frames, a_frames)  
    agent_to_human = build_human_to_agent_match(h_frames, a_frames) 


    # Agent→Human (agent resampled to human timeline)
    a_lv_on_h = [a_lv[i] for i in human_to_agent]
    a_la_on_h = [a_la[i] for i in human_to_agent]
    a_lp_on_h = [a_lp[i] for i in human_to_agent]

    # Agent→Human alignment for all relative series
    a_rel_on_h = {
        k: {m: [a_rel[k][m][i] for i in human_to_agent] for m in rel_modes}
        for k in kinds
    }

    # Human→Agent (human resampled to agent timeline)
    h_lv_on_a = remap_by_index(h_lv, agent_to_human, len(a_frames))
    h_la_on_a = remap_by_index(h_la, agent_to_human, len(a_frames))
    h_lp_on_a = remap_by_index(h_lp, agent_to_human, len(a_frames))

    # actions
    h_actions_all, h_action_times = infer_actions(h_frames)
    a_actions_all, a_action_times = infer_actions(a_frames, agent_mode=True)
    a_actions_on_h = [a_actions_all[i] for i in human_to_agent]
    h_actions_on_a = [h_actions_all[j] for j in agent_to_human]

    # time axes (seconds if available)
    xh_sec = get_time_axis(h_frames)
    xa_sec = get_time_axis(a_frames)

    # Plot
    T = max(len(h_lv), len(a_lv))
    xh = np.arange(len(h_lv)); xa = np.arange(len(a_lv))
    fig, axs = plt.subplots(4, 1, figsize=(12, 36), sharex=False)

    # x-axes
    xh = np.linspace(0, 1, len(h_lv))            # human raw, normalized
    xa = np.linspace(0, 1, len(a_lv))            # agent raw, normalized
    xh_align = np.linspace(0, 1, len(a_lv_on_h)) # human timeline (agent→human uses xh)
    xa_align = np.linspace(0, 1, len(h_lv_on_a)) # agent timeline (human→agent uses xa)

    def triple_plot(ax, h_raw, a_raw, a_on_h, h_on_a, ylabel):
        # raw (non-remapped)
        ax.plot(xh, h_raw, color='#e07a7a', lw=1.6, label='Human (raw)')
        ax.plot(xa, a_raw, color='#1b9e77', lw=1.6, label='Agent (raw)')
        # Agent→Human (dashed, uses human x)
        ax.plot(xh, a_on_h, color='#1b9e77', lw=1.2, ls='--', label='Agent→Human')
        # Human→Agent (dotted, uses agent x)
        ax.plot(xa, h_on_a, color='#e07a7a', lw=1.2, ls=':',  label='Human→Agent')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    triple_plot(axs[0], h_lv, a_lv, a_lv_on_h, h_lv_on_a, 'log visual @ target')
    triple_plot(axs[1], h_la, a_la, a_la_on_h, h_la_on_a, 'log audio @ target')
    triple_plot(axs[2], h_lp, a_lp, a_lp_on_h, h_lp_on_a, 'log posterior @ target')

    # actions (raw, normalized x)
    _action_y = {'stay':0,'turn_left':1,'turn_right':1.3,'step_forward':2,'commit':2.6}
    def _scatter_actions(ax, acts, xs, color, offset=0.0):
        ys = [_action_y.get(a,0)+offset for a in acts]
        ax.scatter(xs, ys, s=26, color=color, alpha=0.7)

    def _draw_world_snapshot(ax, scene_data, h_step, a_step, title=None):
        draw_episode_like_maps(ax, scene_data, show_grid=False, tiny=True, agent_override=None, draw_agent=False)
        def arrow(step, color, style):
            try:
                ax_u, az_u = float(step['position']['x']), float(step['position']['z'])
                gx, gy = unity_to_grid(ax_u, az_u)
                th = np.deg2rad(float(step['rotation'].get('y', 0.0))); L = 2.0
                dx, dy = L*np.sin(th), -L*np.cos(th)
                ax.plot([gx],[gy], marker='o', markersize=3.0, color=color, zorder=5)
                ax.arrow(gx, gy, dx, dy, head_width=0.45, head_length=0.6,
                        length_includes_head=True, color=color, linestyle=style, linewidth=1.0, zorder=5)
            except Exception:
                pass
        arrow(h_step, '#e07a7a', '-'); arrow(a_step, '#1b9e77', '--')
        if title: ax.set_title(title, fontsize=8)

    def _add_minimap_strip(
        fig,
        scene,
        pairs,
        *,
        # Optional overlay rows: list of tuples (title, human_maps, agent_maps, human_cmap, agent_cmap, human_alpha, agent_alpha)
        overlays=None,
        max_k=10,
        # layout
        bottom=0.05,            # bottom of the whole block
        row_h=0.18,             # height per row
        vgap=0.002,             # vertical gap between rows
        left_margin=0.06,
        right_margin=0.98,
        gap=0.01
    ):
        """
        Draw a block of minimaps with 1 + len(overlays) rows.
        - Cols = min(max_k, len(pairs)); one column per (human_idx, agent_idx).
        - Row 1: world-only (human red, agent green).
        - Overlay rows (if provided): world + heatmap overlays for both human and matched agent.

        'pairs' is a list of (human_idx, agent_idx) indices.
        """
        # helper: world-only with arrows
        def _draw_world_only(ax, h_step, a_step, title=None):
            draw_episode_like_maps(ax, scene, show_grid=False, tiny=True, agent_override=None, draw_agent=False)

            
            def _arrow(step, color, style, edgecolor='k', marker='o'):
                try:
                    ux, uz = float(step['position']['x']), float(step['position']['z'])
                    gx, gy = unity_to_grid(ux, uz)
                    th = np.deg2rad(float(step['rotation'].get('y', 0.0)))
                    L = 2.0
                    dx, dy = L*np.sin(th), -L*np.cos(th)

                    # Draw point marker
                    ax.plot([gx], [gy], marker=marker, markersize=3.0, color=color, zorder=5)

                    # 1) Draw the shaft in the desired linestyle
                    ax.plot([gx, gx + dx], [gy, gy + dy],
                            color=color, linestyle=style, linewidth=1.0, zorder=5)

                    # 2) Draw a separate *solid* arrowhead
                    head_len = 0.6
                    head_wid = 0.45

                    ax.arrow(
                        gx + dx * (1 - head_len / L),
                        gy + dy * (1 - head_len / L),
                        dx * (head_len / L),
                        dy * (head_len / L),
                        head_width=head_wid,
                        head_length=head_len,
                        color=color,
                        linestyle='solid',       # force solid head
                        linewidth=1.0,
                        length_includes_head=True,
                        zorder=6                 # draw above shaft
                    )

                except Exception:
                    pass
# 
            _arrow(a_step, '#1b9e77', style='solid', edgecolor=None, marker='o') # Agent arrow
            _arrow(h_step, '#e07a7a', style='--', edgecolor='k', marker='x') # Human arrow
            # if title: ax.set_title(title, fontsize=8)

        # helper: world + two overlays (human + agent)
        def _draw_overlay(ax, h_step, a_step, h_map, a_map, title, h_cmap, a_cmap, h_alpha, a_alpha, show_cb=False):
            draw_episode_like_maps(ax, scene, show_grid=False, tiny=True, agent_override=h_step, draw_agent=False)
            # if h_map is not None:
            #     overlay_belief_on_world(ax, h_step, h_map, cmap=h_cmap, alpha=0.1, blur_sigma=(0.7, 1.5), upsample=3, point_size=6, agent_color='#e07a7a')
            if a_map is not None:
                # overlay_belief_on_world(ax, a_step, a_map, cmap=a_cmap, alpha=0.1, blur_sigma=(0.7, 1.5), upsample=3, point_size=6, agent_color='#1b9e77')
                overlay_belief_on_world(ax, a_step, a_map, cmap=a_cmap, alpha=0.1, blur_sigma=(0.0, 0.0), upsample=3, point_size=6, agent_color='#1b9e77', draw_colorbar=show_cb)
            # ax.set_title(title, fontsize=7)

        # columns to show
        pick = pairs[:max_k]
        C = len(pick)
        if C == 0:
            return

        total_w = (right_margin - left_margin)
        ax_w = (total_w - gap*(C-1)) / C

        # Number of rows: 1 world row + N overlay rows
        overlay_rows = overlays or []
        R = 1 + len(overlay_rows)

        # Top of the block = bottom + (R*row_h + (R-1)*vgap)
        # We create rows from top to bottom so titles sit above each panel comfortably
        def _row_bottom(r_from_top):
            # r_from_top = 0 for top row, R-1 for bottom row
            top_of_block = bottom + (R*row_h + (R-1)*vgap)
            return top_of_block - (r_from_top+1)*row_h - r_from_top*vgap

        # draw each column
        for col, (hj, ai) in enumerate(pick):
            # safe indices
            hj = int(hj)
            ai = int(ai) if (ai is not None and ai >= 0) else None
            h_step = h_frames[hj]
            a_step = a_frames[ai] if ai is not None and ai < len(a_frames) else a_frames[-1]

            left = left_margin + col*(ax_w + gap)

            # row 0 (top): world only
            y0 = _row_bottom(0)
            ax_world = fig.add_axes([left, y0, ax_w, row_h])
            _draw_world_only(ax_world, h_step, a_step, title=f'A{ai if ai is not None else "?"}/H{hj}')

            # overlay rows
            for r, (title, h_maps, a_maps, h_cmap, a_cmap, h_alpha, a_alpha) in enumerate(overlay_rows, start=1):
                y = _row_bottom(r)
                ax = fig.add_axes([left, y, ax_w, row_h])
                h_map = (h_maps[hj] if 0 <= hj < len(h_maps) else None) if h_maps is not None else None
                a_map = (a_maps[ai] if (ai is not None and 0 <= ai < len(a_maps)) else None) if a_maps is not None else None
                show_cb = True # (col == C - 1)
                _draw_overlay(ax, h_step, a_step, h_map, a_map, title, h_cmap, a_cmap, h_alpha, a_alpha, show_cb=show_cb)


    def save_raw_plot(out_path):
        fig, axs = plt.subplots(4,1, figsize=(12,20), sharex=False)
        

        # minimaps: sample along AGENT steps, pair to nearest human
        idx_agent = np.unique(np.linspace(0, len(a_frames)-1, num=min(10,len(a_frames)), dtype=int))
        pairs = [(agent_to_human[i], i) for i in idx_agent]
        plt.subplots_adjust(
            left=0.08, right=0.98, top=0.96,
            bottom=0.32,   # <- raise this from 0.26 to 0.32 or more
            hspace=0.1     # slightly more vertical spacing between ax[3] and the others
        )
        # how many overlay rows you'll draw (visual, audio, posterior) + 1 world row
        overlay_rows = 3
        rows_total = 1 + overlay_rows

        # choose sizes; you can shrink these if needed
        row_h = 0.12   # was 0.18
        vgap  = 0.003
        bottom_strip = 0.04

        # IMPORTANT: lift the main plots *before* adding the minimaps
        _reserve_space_for_minimaps(rows=rows_total, row_h=row_h, vgap=vgap, bottom=bottom_strip)

        # now add the minimaps using the same geometry so the block fits exactly
        _add_minimap_strip(
            axs[0].figure, scene, pairs, max_k=10,
            bottom=bottom_strip, row_h=row_h, vgap=vgap,
            left_margin=0.02, right_margin=1.0, gap=0.0,
            overlays=[
                ("audio",     h_la_maps, a_la_maps, 'Oranges', 'Oranges', 0.45, 0.35),
                ("visual",    h_lv_maps, a_lv_maps, 'Blues',   'Blues',  0.45, 0.35),
                ("posterior", h_lp_maps, a_lp_maps, 'viridis', 'magma',   0.45, 0.35),
            ],
        )

        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f'[ok] saved {out_path}')

    def save_a2h_plot(out_path):
        fig, axs = plt.subplots(4,1, figsize=(12,20), sharex=False)
        # use HUMAN timeline (index -> optionally seconds if present)
        # xh = get_time_axis(h_frames)
        # time axes
        xh = get_time_axis(h_frames)   # human seconds (or indices)
        xa = get_time_axis(a_frames)   # agent seconds (or indices)
        axs[0].plot(xh, h_lv,      color='#e07a7a', lw=1.6, label='Human (raw)')
        axs[0].plot(xh, a_lv_on_h, color='#1b9e77', lw=1.6, label='Agent→Human')
        axs[0].set_ylabel('log visual @ target'); axs[0].grid(True, alpha=0.3); axs[0].legend(loc='upper left')

        axs[1].plot(xh, h_la,      color='#e07a7a', lw=1.6)
        axs[1].plot(xh, a_la_on_h, color='#1b9e77', lw=1.6)
        axs[1].set_ylabel('log audio @ target'); axs[1].grid(True, alpha=0.3)

        axs[2].plot(xh, h_lp,      color='#e07a7a', lw=1.6)
        axs[2].plot(xh, a_lp_on_h, color='#1b9e77', lw=1.6)
        axs[2].set_ylabel('log posterior @ target'); axs[2].grid(True, alpha=0.3)


        # --- bottom panel: actions on human timeline, but agent points are NOT densified ---
        # human: one marker per human step at xh
        _scatter_actions_at(axs[3], h_actions_all, xh[:len(h_actions_all)], '#e07a7a', 0.05)

        # agent: map each AGENT step i → nearest HUMAN step j, plot at xh[j]
        a2h_map = build_human_to_agent_match(h_frames, a_frames)  # len == len(a_frames)
        x_agent_on_h = np.array([ xh[j] for j in a2h_map ], dtype=float)
        _scatter_actions_at(axs[3], a_actions_all, x_agent_on_h[:len(a_actions_all)], '#1b9e77', -0.05)

        axs[3].set_yticks([0,1,1.3,2,2.6])
        axs[3].set_yticklabels(['stay','turn_L','turn_R','forward','commit'])
        axs[3].set_xlabel('human time (s)' if (len(xh)>1 and (xh[-1]-xh[0])>0) else 'human step')
        axs[3].grid(True, axis='x', alpha=0.3)


        # minimaps: sample along HUMAN steps, pair to nearest agent
        idx_human = np.unique(np.linspace(0, len(h_frames)-1, num=min(10,len(h_frames)), dtype=int))
        pairs = [(j, human_to_agent[j]) for j in idx_human]
        # plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.26, hspace=0.35)
        plt.subplots_adjust(
            left=0.08, right=0.98, top=0.96,
            bottom=0.32,   # <- raise this from 0.26 to 0.32 or more
            hspace=0.4     # slightly more vertical spacing between ax[3] and the others
        )
       # how many overlay rows you'll draw (visual, audio, posterior) + 1 world row
        overlay_rows = 3
        rows_total = 1 + overlay_rows

        # choose sizes; you can shrink these if needed
        row_h = 0.12   # was 0.18
        vgap  = 0.010
        bottom_strip = 0.04

        # IMPORTANT: lift the main plots *before* adding the minimaps
        _reserve_space_for_minimaps(rows=rows_total, row_h=row_h, vgap=vgap, bottom=bottom_strip)

        # now add the minimaps using the same geometry so the block fits exactly
        _add_minimap_strip(
            axs[0].figure, scene, pairs, max_k=10,
            bottom=bottom_strip, row_h=row_h, vgap=vgap,
            left_margin=0.06, right_margin=0.98, gap=0.02,
            overlays=[
                ("visual",    h_lv_maps, a_lv_maps, 'Blues',   'Greens',  0.45, 0.35),
                ("audio",     h_la_maps, a_la_maps, 'Oranges', 'Purples', 0.45, 0.35),
                ("posterior", h_lp_maps, a_lp_maps, 'viridis', 'magma',   0.45, 0.35),
            ],
        )

        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=200); plt.close(fig)
        print(f'[ok] saved {out_path}')

    def save_h2a_plot(out_path):
        fig, axs = plt.subplots(4,1, figsize=(12,20), sharex=False)
        # use AGENT timeline (seconds if present)
        xa = get_time_axis(a_frames)
        axs[0].plot(xa, a_lv,      color='#1b9e77', lw=1.6, label='Agent (raw)')
        axs[0].plot(xa, h_lv_on_a, color='#e07a7a', lw=1.6, label='Human→Agent')
        axs[0].set_ylabel('log visual @ target'); axs[0].grid(True, alpha=0.3); axs[0].legend(loc='upper left')

        axs[1].plot(xa, a_la,      color='#1b9e77', lw=1.6)
        axs[1].plot(xa, h_la_on_a, color='#e07a7a', lw=1.6)
        axs[1].set_ylabel('log audio @ target'); axs[1].grid(True, alpha=0.3)

        axs[2].plot(xa, a_lp,      color='#1b9e77', lw=1.6)
        axs[2].plot(xa, h_lp_on_a, color='#e07a7a', lw=1.6)
        axs[2].set_ylabel('log posterior @ target'); axs[2].grid(True, alpha=0.3)

        _scatter_actions(axs[3], a_actions_all, np.linspace(0,1,len(a_actions_all)), '#1b9e77', -0.05)
        _scatter_actions(axs[3], h_actions_on_a, np.linspace(0,1,len(h_actions_on_a)), '#e07a7a', 0.05)
        axs[3].set_yticks([0,1,1.3,2,2.6]); axs[3].set_yticklabels(['stay','turn_L','turn_R','forward','commit'])
        axs[3].set_xlabel('agent time (s)' if len(xa)>1 and xa.max()>0 else 'agent step'); axs[3].grid(True, axis='x', alpha=0.3)

        # minimaps: sample along AGENT steps, pair to nearest human
        idx_agent = np.unique(np.linspace(0, len(a_frames)-1, num=min(10,len(a_frames)), dtype=int))
        pairs = [(agent_to_human[i], i) for i in idx_agent]
        plt.subplots_adjust(
            left=0.08, right=0.98, top=0.96,
            bottom=0.32,   # <- raise this from 0.26 to 0.32 or more
            hspace=0.4     # slightly more vertical spacing between ax[3] and the others
        )
        # how many overlay rows you'll draw (visual, audio, posterior) + 1 world row
        overlay_rows = 3
        rows_total = 1 + overlay_rows

        # choose sizes; you can shrink these if needed
        row_h = 0.12   # was 0.18
        vgap  = 0.010
        bottom_strip = 0.04

        # IMPORTANT: lift the main plots *before* adding the minimaps
        _reserve_space_for_minimaps(rows=rows_total, row_h=row_h, vgap=vgap, bottom=bottom_strip)

        # now add the minimaps using the same geometry so the block fits exactly
        _add_minimap_strip(
            axs[0].figure, scene, pairs, max_k=10,
            bottom=bottom_strip, row_h=row_h, vgap=vgap,
            left_margin=0.06, right_margin=0.98, gap=0.02,
            overlays=[
                ("visual",    h_lv_maps, a_lv_maps, 'Blues',   'Greens',  0.45, 0.35),
                ("audio",     h_la_maps, a_la_maps, 'Oranges', 'Purples', 0.45, 0.35),
                ("posterior", h_lp_maps, a_lp_maps, 'viridis', 'magma',   0.45, 0.35),
            ],
        )

        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=200); plt.close(fig)
        print(f'[ok] saved {out_path}')

    def save_a2h_plot_rel(out_path, mode):
        # Full A→H plot, but rows 0–2 are relative advantage for visual/audio/posterior (given `mode`)
        fig, axs = plt.subplots(4,1, figsize=(12,20), sharex=False)

        # x-axis = human time (seconds if available, else index)
        xh = get_time_axis(h_frames)

        def _row(ax, kind, label):
            ax.plot(xh, h_rel[kind][mode],            color='#e07a7a', lw=1.6, label='Human (raw)')
            ax.plot(xh, a_rel_on_h[kind][mode],       color='#1b9e77', lw=1.6, label='Agent→Human')
            ax.axhline(0, color='k', lw=1, alpha=0.35)
            ax.set_ylabel(f'rel log-adv {label} ({mode})')
            ax.grid(True, alpha=0.3)

        # rows 0–2: relative series for visual, audio, posterior
        _row(axs[0], "visual",    "visual")
        _row(axs[1], "audio",     "audio")
        _row(axs[2], "posterior", "posterior")

        # row 3: actions on human timeline (same logic you already use)
        _scatter_actions_at(axs[3], h_actions_all, xh[:len(h_actions_all)], '#e07a7a', 0.05)
        a2h_map = build_human_to_agent_match(h_frames, a_frames)
        x_agent_on_h = np.array([ xh[j] for j in a2h_map ], dtype=float)
        _scatter_actions_at(axs[3], a_actions_all, x_agent_on_h[:len(a_actions_all)], '#1b9e77', -0.05)
        axs[3].set_yticks([0,1,1.3,2,2.6])
        axs[3].set_yticklabels(['stay','turn_L','turn_R','forward','commit'])
        axs[3].set_xlabel('human time (s)' if (len(xh)>1 and (xh[-1]-xh[0])>0) else 'human step')
        axs[3].grid(True, axis='x', alpha=0.3)

        # ---- minimap block (unchanged overlays; we’re swapping only the top 3 rows) ----
        idx_human = np.unique(np.linspace(0, len(h_frames)-1, num=min(10,len(h_frames)), dtype=int))
        pairs = [(j, human_to_agent[j]) for j in idx_human]

        # lift the main stacks so minimaps won’t overlap
        overlay_rows = 3
        rows_total = 1 + overlay_rows
        row_h = 0.12
        vgap  = 0.010
        bottom_strip = 0.04
        _reserve_space_for_minimaps(rows=rows_total, row_h=row_h, vgap=vgap, bottom=bottom_strip)

        # keep original overlays (raw log maps) beneath
        _add_minimap_strip(
            axs[0].figure, scene, pairs, max_k=10,
            bottom=bottom_strip, row_h=row_h, vgap=vgap,
            left_margin=0.06, right_margin=0.98, gap=0.02,
            overlays=[
                ("visual",    h_lv_maps, a_lv_maps, 'Blues',   'Greens',  0.45, 0.35),
                ("audio",     h_la_maps, a_la_maps, 'Oranges', 'Purples', 0.45, 0.35),
                ("posterior", h_lp_maps, a_lp_maps, 'viridis', 'magma',   0.45, 0.35),
            ],
        )

        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=200); plt.close(fig)
        print(f'[ok] saved {out_path}')

    base = os.path.splitext(out_path)[0]
    save_raw_plot(base + "_raw.png")
    save_a2h_plot(base + "_a2h.png")
    save_h2a_plot(base + "_h2a.png")

    plot_entropy_eig_relationships(
        base + "_entropy_eig.png",
        human={
            "post_maps": h_lp_maps,
            "prior_maps": h_lprior_maps,
            "actions": h_actions_all,
            "times": h_action_times,
            "label": "Human",
        },
        agent={
            "post_maps": a_lp_maps,
            "prior_maps": a_lprior_maps,
            "actions": a_actions_all,
            "times": a_action_times,
            "label": "Agent",
        },
    )


def main():
    ap = argparse.ArgumentParser(description='Compare human and agent belief/action over time for a map')
    ap.add_argument('--human-root', type=str, default='analysis/data', help='Root directory containing human p01..p12')
    ap.add_argument('--pid', type=str, required=True, help='Participant id (e.g., p04)')
    ap.add_argument('--agent-root', type=str, required=True, help='Root directory containing agent_* subfolders for a model')
    ap.add_argument('--agent-id', type=str, default='agent_1', help='Agent id subfolder (e.g., agent_1)')
    ap.add_argument('--map-id', type=str, required=True, help='Map id to plot (e.g., M0086)')
    ap.add_argument('--maps-json', type=str, default='maps/maps.json', help='maps.json for map→condition lookup')
    ap.add_argument('--out', type=str, default='analysis/plots/belief_action/{pid}_{agent}_{map}.png')
    ap.add_argument('--audio-like-source', type=str, default='geometric', choices=['wav','geometric'],
                    help="How to compute audio log-likelihood in belief_action_comparison: 'wav' uses compute_audio_log_like; 'geometric' uses a simple geometric model.")
    ap.add_argument('--agent-belief', type=str, default='recompute',
                choices=['saved','recompute'],
                help="Use agent's saved log maps or recompute from pose/scene like human.")
    args = ap.parse_args()

    h_ep = find_human_episode_dir(args.human_root, args.pid, args.map_id)
    a_ep = find_agent_trial_dir(args.agent_root, args.agent_id, args.map_id)
    if not h_ep or not a_ep:
        print(f'[error] episode not found. human={h_ep}, agent={a_ep}')
        return
    out_path = args.out.format(pid=args.pid, agent=args.agent_id, map=args.map_id)
    try:
        from scripts.test_models_and_analyze import _load_maps_conditions
        map_conditions = _load_maps_conditions(args.maps_json) if args.maps_json else {}
    except Exception as e:
        print(f"[warn] failed to load map conditions from {args.maps_json}: {e}")
        map_conditions = {}

    plot_belief_action(
        h_ep,
        a_ep,
        out_path,
        audio_like_source=args.audio_like_source,
        agent_belief_mode=args.agent_belief,
    )

    human_all, agent_all, human_by_map, agent_by_map = gather_total_distances_by_map(
        args.human_root,
        args.agent_root,
    )
    map_ids_for_panels = sorted(
        set(map_conditions.keys()) | set(human_by_map.keys()) | set(agent_by_map.keys())
    )

    plot_distance_histogram(
        "analysis/plots/distance_histogram.png",
        human_root=args.human_root,
        agent_root=args.agent_root,
        human_dists=human_all,
        agent_dists=agent_all,
    )
    plot_distance_histogram_by_map(
        "analysis/plots/distance_histogram_by_map.png",
        map_ids=map_ids_for_panels,
        human_by_map=human_by_map,
        agent_by_map=agent_by_map,
    )
    plot_distance_histogram_by_condition(
        "analysis/plots/distance_histogram_by_condition.png",
        map_ids=map_ids_for_panels,
        map_conditions=map_conditions,
        human_by_map=human_by_map,
        agent_by_map=agent_by_map,
    )



if __name__ == '__main__':
    main()
