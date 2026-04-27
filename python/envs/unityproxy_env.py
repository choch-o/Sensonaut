import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import math 
import numpy as np
from collections import deque
from typing import List, Dict, Tuple
import wandb
from utils.constants import THETA_GRID, R_GRID
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.unity_client import UnityControlClient  # Assuming this contains your provided code
from utils.audio_features import compute_itd_ild_from_wav  # Import the audio feature extraction function
import json
import os
from datetime import datetime

class UnityProxyEnv(gym.Env):
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

    def _rotate_beliefs_by(self, delta_theta: float) -> None:
        """Rotate cached distributions into the new head-relative frame."""
        if hasattr(self, "_last_posterior"):
            self._last_posterior = self._roll_theta(self._last_posterior, delta_theta)
        if hasattr(self, "_last_audio_like"):
            self._last_audio_like = self._roll_theta(self._last_audio_like, delta_theta)
        if hasattr(self, "_last_visual_like"):
            self._last_visual_like = self._roll_theta(self._last_visual_like, delta_theta)
        if hasattr(self, "_last_log_visual"):
            self._last_log_visual = self._roll_theta(self._last_log_visual, delta_theta)
        if hasattr(self, "_last_log_post"):
            self._last_log_post = self._roll_theta(self._last_log_post, delta_theta)
        if hasattr(self, "_last_log_audio"):
            self._last_log_audio = self._roll_theta(self._last_log_audio, delta_theta)
        # Keep internal marker in sync to avoid double-rotation inside _update_estimate
        self._last_heading = self.agent_heading

    # Action mapping dictionary
    ACTION_NAMES = {
        0: "turn_left",
        1: "turn_right", 
        2: "move_forward",
        3: "commit",
        4: "stay"
    }

    def __init__(
        self,
        render_mode: str = "rgb_array",
        grid_w: int = 13,
        grid_h: int = 29,
        n_objects: int = 12,
        n_slots: int = 16,
        max_steps: int = 20,
        theta_error_bound: float = 0.2,
        r_error_bound: float = 0.5,
        step_penalty: float = 0.0,
        forward_penalty: float = 0.3,
        turn_penalty: float = 0.1,
        collision_penalty: float = 5.0,
        fov_angle: float = math.radians(120),
        wandb_run = None,
        history_length: int = 4,
        alpha: float = 0.3,  # weight of new observations in Bayesian filtering
        visual_alpha: float = 0.5,  # weight of visual observations in Bayesian filtering
        lambda_color: float = 0.1,  # color confusion parameter
        mode: str = "train",
        head_radius: float = 0.0875,
        speed_of_sound: float = 343.0,
        itd_noise_scale: float = 30.0,
        ild_noise_scale: float = 0.1,
        visual_audio_ratio: float = 0.7,
        sigma_theta_deg: float = 15.0,
        # Visual bearing/range noise controls
        sigma_r_min: float = 0.25,
        alpha_r: float = 0.12,
        turn_angle: float = 90.0,  # degrees
        pace: float = 1.0,
        visual_exclusion_decay: float = 0.3,
        visible_weight: float = 5.0,
        debug_polar: bool = False,
        use_posterior_obs: bool = False,
        use_unity: bool = True,
        test_map_pid: str = 'p01',
        correct_object: bool = True,
        # Reward shaping: bonus for committing early (more time left)
        time_bonus_weight: float = 0.0,
        time_bonus_success_only: bool = True,
    ) -> None:
        super().__init__()
        self.wandb_run = wandb_run
        self.render_mode = render_mode
        self.history_length = history_length
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.obs_theta_history = deque(maxlen=self.history_length)
        self.obs_r_history = deque(maxlen=self.history_length)
        self.action_history = deque(maxlen=self.history_length)
        self.remaining_steps_history = deque(maxlen=self.history_length)
        self.n_objects = n_objects
        self.n_slots = n_slots
        self.max_steps = max_steps
        self.theta_error_bound = theta_error_bound
        self.r_error_bound = r_error_bound
        self.step_penalty = step_penalty
        self.forward_penalty = forward_penalty
        self.turn_penalty = turn_penalty
        self.collision_penalty = collision_penalty
        self.fov_angle = fov_angle
        self.alpha = alpha
        self.visual_alpha = visual_alpha
        self.lambda_color = lambda_color
        self.turn_angle = turn_angle
        self.pace = pace
        self.mode = mode
        self.use_posterior_obs = use_posterior_obs
        self.test_map_pid = test_map_pid
        self.visible_weight = visible_weight
        self.detected_objects = []
        self.visual_audio_ratio = visual_audio_ratio

        self._last_visual_like = np.ones((R_GRID.shape[0], THETA_GRID.shape[0])) / (R_GRID.shape[0] * THETA_GRID.shape[0])

        # Visual discount strength
        self.visual_exclusion_decay = visual_exclusion_decay  # used for both free space and different-color surfaces

        # Store physics and noise parameters as instance variables
        self.head_radius = head_radius
        self.speed_of_sound = speed_of_sound / 1_000_000  # convert to m/μs if needed
        self.itd_noise_scale = itd_noise_scale
        self.ild_noise_scale = ild_noise_scale
        self.debug_polar = debug_polar
        self.sigma_theta_deg = sigma_theta_deg
        self.sigma_r_min = sigma_r_min
        self.alpha_r = alpha_r

        self.correct_object = correct_object
        # Reward shaping params
        self.time_bonus_weight = float(time_bonus_weight)
        self.time_bonus_success_only = bool(time_bonus_success_only)


        # Action space – discrete {0,1,2}
        self.action_space: gym.Space = spaces.Discrete(5) # 0=left,1=right,2=forward,3=commit,4=stay

        # If use_posterior_obs=True, the observation additionally includes the full posterior grid
        # flattened as a vector and its entropy. This turns the POMDP into a belief-MDP observation.
        base_obs_spaces = {
            "est_theta": spaces.Box(-math.pi, math.pi, shape=(self.history_length,), dtype=np.float32),
            "theta_uncertainty": spaces.Box(0.0, math.pi, shape=(1,), dtype=np.float32),
            "est_r": spaces.Box(0.0, float(max(self.grid_w, self.grid_h)), shape=(self.history_length,), dtype=np.float32),
            "r_uncertainty": spaces.Box(0.0, float(max(self.grid_w, self.grid_h)), shape=(1,), dtype=np.float32),
            "last_actions": spaces.MultiDiscrete([6] * self.history_length),
            "remaining_steps": spaces.Box(0, self.max_steps, shape=(self.history_length,), dtype=np.int32),
        }
        if self.use_posterior_obs:
            # Flattened posterior over (r, theta); values in [0,1], sum≈1
            posterior_size = int(R_GRID.shape[0] * THETA_GRID.shape[0])
            base_obs_spaces["posterior"] = spaces.Box(0.0, 1.0, shape=(posterior_size,), dtype=np.float32)
            # Optional: entropy of posterior (0..log N)
            base_obs_spaces["posterior_entropy"] = spaces.Box(0.0, float(np.log(max(1, posterior_size))), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(base_obs_spaces)

        # Unity client
        self.use_unity = use_unity
        if self.use_unity:
            self.client = UnityControlClient()
            self.client.connect()
        else:
            self.client = None

        # Initialize env state
        self.reset()

    def _ego_from_world(self, wx: int | float, wy: int | float) -> tuple[float, float]:
        """Convert a world (x,y) location to agent-egocentric (r, theta).
        Theta is wrapped to (-pi, pi] with 0 at "ahead" (north in our convention).
        """
        dx = float(wx) - float(self.agent_pos[0])
        dy = float(wy) - float(self.agent_pos[1])
        r = math.hypot(dx, dy)
        theta_abs = math.atan2(dx, -dy) % (2 * math.pi)
        theta_rel = (theta_abs - self.agent_heading + math.pi) % (2 * math.pi) - math.pi
        return r, theta_rel
    
    def log_episode_data(self, episode_num: int = None):
        """
        Log complete episode data including map info, state/observation sequence, 
        actions, rewards, and movement statistics to a JSON file in the logs directory.
        
        Args:
            episode_num: Optional episode number
        """
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate timestamp and filename
        timestamp = datetime.now().isoformat()
        filename = f"episode_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{episode_num if episode_num is not None else 'unknown'}.json"
        filepath = os.path.join(logs_dir, filename)
        
        # Prepare complete episode data
        episode_data = {
            "timestamp": timestamp,
            "episode": episode_num,
            "total_steps": len(getattr(self, 'episode_states', [])),
            "episode_outcome": {
                "success": getattr(self, 'last_episode_success', None),
                "final_reward": getattr(self, 'last_episode_final_reward', None),
                "total_reward": sum(getattr(self, 'episode_rewards', [])),
                "steps_taken": getattr(self, 'steps_taken', 0),
                "head_turns": getattr(self, 'head_turns', 0),
                "locomotion_steps": getattr(self, 'locomotion_steps', 0)
            },
            "map_info": {
                "grid_dimensions": {
                    "width": self.grid_w,
                    "height": self.grid_h
                },
                "initial_agent": {
                    "position": {
                        "x": float(getattr(self, 'initial_agent_pos', [0, 0])[0]),
                        "y": float(getattr(self, 'initial_agent_pos', [0, 0])[1])
                    },
                    "heading": float(getattr(self, 'initial_agent_heading', 0.0)),
                    "heading_degrees": float(math.degrees(getattr(self, 'initial_agent_heading', 0.0)))
                },
                "objects": [],
                "target_idx": getattr(self, 'target_idx', None)
            },
            "environment_config": {
                "n_objects": self.n_objects,
                "n_slots": self.n_slots,
                "max_steps": self.max_steps,
                "turn_angle": self.turn_angle,
                "mode": self.mode,
                "theta_error_bound": self.theta_error_bound,
                "r_error_bound": self.r_error_bound,
                "forward_penalty": self.forward_penalty,
                "turn_penalty": self.turn_penalty,
                "collision_penalty": self.collision_penalty
            },
            "episode_sequence": []
        }
        
        # Add objects information if available
        if hasattr(self, 'objects') and self.objects:
            for i, obj in enumerate(self.objects):
                obj_info = {
                    "id": i,
                    "position": {
                        "x": float(obj["pos"][0]),
                        "y": float(obj["pos"][1])
                    },
                    "color": int(obj["color"]),
                    "is_target": (i == getattr(self, 'target_idx', -1))
                }
                episode_data["map_info"]["objects"].append(obj_info)
        
        # Add episode sequence data
        episode_states = getattr(self, 'episode_states', [])
        episode_actions = getattr(self, 'episode_actions', [])
        episode_rewards = getattr(self, 'episode_rewards', [])
        
        for step in range(len(episode_states)):
            action_num = episode_actions[step] if step < len(episode_actions) else None
            step_data = {
                "step": step,
                "state": {
                    "agent_position": {
                        "x": float(episode_states[step]["agent_pos"][0]),
                        "y": float(episode_states[step]["agent_pos"][1])
                    },
                    "agent_heading": float(episode_states[step]["agent_heading"]),
                    "agent_heading_degrees": float(math.degrees(episode_states[step]["agent_heading"])),
                    "estimates": {
                        "est_theta": float(episode_states[step]["est_theta"]),
                        "est_r": float(episode_states[step]["est_r"]),
                        "theta_uncertainty": float(episode_states[step]["theta_uncertainty"]),
                        "r_uncertainty": float(episode_states[step]["r_uncertainty"])
                    }
                },
                "action": {
                    "id": int(action_num) if action_num is not None else None,
                    "name": self.ACTION_NAMES.get(int(action_num), "unknown") if action_num is not None else None
                },
                "reward": float(episode_rewards[step]) if step < len(episode_rewards) else None
            }
            episode_data["episode_sequence"].append(step_data)
        
        # Write to JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(episode_data, f, indent=2)
            print(f"Episode data logged to: {filepath}")
        except Exception as e:
            print(f"Failed to log episode data: {e}")
        
        return filepath

    def log_map_info(self, episode_num: int = None, step_num: int = None):
        """
        Log map information with timestamp to a JSON file in the logs directory.
        
        Args:
            episode_num: Optional episode number
            step_num: Optional step number
        """
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate timestamp and filename
        timestamp = datetime.now().isoformat()
        filename = f"map_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(logs_dir, filename)
        
        # Prepare map information
        map_info = {
            "timestamp": timestamp,
            "episode": episode_num,
            "step": step_num,
            "grid_dimensions": {
                "width": self.grid_w,
                "height": self.grid_h
            },
            "agent": {
                "position": {
                    "x": float(self.agent_pos[0]) if hasattr(self, 'agent_pos') else None,
                    "y": float(self.agent_pos[1]) if hasattr(self, 'agent_pos') else None
                },
                "heading": float(self.agent_heading) if hasattr(self, 'agent_heading') else None,
                "heading_degrees": float(math.degrees(self.agent_heading)) if hasattr(self, 'agent_heading') else None
            },
            "objects": [],
            "target_idx": getattr(self, 'target_idx', None),
            "environment_config": {
                "n_objects": self.n_objects,
                "n_slots": self.n_slots,
                "max_steps": self.max_steps,
                "turn_angle": self.turn_angle,
                "mode": self.mode,
                "theta_error_bound": self.theta_error_bound,
                "r_error_bound": self.r_error_bound
            },
            "estimates": {
                "est_theta": float(self.est_theta) if hasattr(self, 'est_theta') else None,
                "est_r": float(self.est_r) if hasattr(self, 'est_r') else None,
                "theta_uncertainty": float(self.theta_uncertainty) if hasattr(self, 'theta_uncertainty') else None,
                "r_uncertainty": float(self.r_uncertainty) if hasattr(self, 'r_uncertainty') else None
            }
        }
        
        # Add objects information if available
        if hasattr(self, 'objects') and self.objects:
            for i, obj in enumerate(self.objects):
                obj_info = {
                    "id": i,
                    "position": {
                        "x": float(obj["pos"][0]),
                        "y": float(obj["pos"][1])
                    },
                    "color": int(obj["color"]),
                    "is_target": (i == getattr(self, 'target_idx', -1))
                }
                map_info["objects"].append(obj_info)
        
        # Write to JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(map_info, f, indent=2)
            print(f"Map info logged to: {filepath}")
        except Exception as e:
            print(f"Failed to log map info: {e}")
        
        return filepath

    def _grid_to_unity(self, gx: float, gy: float,
                        x_min: float = 25.0, x_max: float = 38.0,
                        z_min: float = -14.5, z_max: float = 14.5) -> tuple[float, float]:
        """Map grid indices (gx, gy) to Unity world coordinates (x, z).
        Top-left grid cell (0,0) → (x_min, z_max); bottom-right (grid_w-1, grid_h-1) → (x_max, z_min).
        """
        # Guard against division by zero for degenerate grids
        gw = max(1, self.grid_w - 1)
        gh = max(1, self.grid_h - 1)
        # Normalize to [0,1]
        nx = gx / gw
        ny = gy / gh
        # Linear map
        x = x_min + nx * (x_max - x_min)
        z = z_max + ny * (z_min - z_max)  # note: decreases with y
        return float(x), float(z)

    # ------------------------------------------------------------------
    #  Snapshots for logging (authoritative from env / Unity)
    # ------------------------------------------------------------------
    def _unity_agent_pose(self) -> tuple[dict, dict]:
        """Return (position, rotation) in Unity coordinates if available; else fallback."""
        # Prefer Unity client pose if available
        try:
            if self.use_unity and self.client is not None:
                vis = self.client.get_visual_observation()
                if vis and "agentPosition" in vis and "agentRotation" in vis:
                    pos = vis["agentPosition"]; rot = vis["agentRotation"]
                    # Keep last observation for step snapshots
                    self._last_visual_obs = vis
                    return (
                        {"x": float(pos.get("x", 0.0)), "y": float(pos.get("y", 0.0)), "z": float(pos.get("z", 0.0))},
                        {"x": float(rot.get("x", 0.0)), "y": float(rot.get("y", 0.0)), "z": float(rot.get("z", 0.0))}
                    )
        except Exception:
            pass
        # Fallback to grid-to-unity conversion
        try:
            ux, uz = self._grid_to_unity(self.agent_pos[0], self.agent_pos[1])
            return (
                {"x": ux, "y": 0.0, "z": uz},
                {"x": 0.0, "y": float(math.degrees(getattr(self, "agent_heading", 0.0))), "z": 0.0}
            )
        except Exception:
            return (
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 0.0, "z": 0.0}
            )

    def get_scene_snapshot(self) -> dict:
        """Build a scene snapshot (vehicles + agent) using current env state and Unity-space mapping."""
        vehicles_payload = []
        for i, obj in enumerate(getattr(self, "objects", [])):
            try:
                gx, gy = int(obj["pos"][0]), int(obj["pos"][1])
                ux, uz = self._grid_to_unity(gx, gy)
                prefab_idx = int(obj.get("color", 0)) % 3
                prefab_name = ["Black", "Red", "White"][prefab_idx]
                vehicles_payload.append({
                    "name": f"{prefab_name}_Car_{i}" + (" (Target)" if i == getattr(self, "target_idx", -1) else ""),
                    "position": {"x": float(ux), "y": 0.6, "z": float(uz)},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                })
            except Exception:
                continue
        pos, rot = self._unity_agent_pose()
        agent_payload = {
            "name": "Sensonaut",
            "position": pos,
            "rotation": rot,
            "map_pid": self.test_map_pid,
            "map_id": self.map_id,
        }
        return {
            "recordTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "totalVehicles": len(vehicles_payload),
            "vehicles": vehicles_payload,
            "agent": agent_payload
        }

    def get_step_snapshot(self) -> dict:
        """Return per-step snapshot including Unity visual observation (if any), agent pose, and latest audio file."""
        pos, rot = self._unity_agent_pose()

        world_xy = self._estimate_to_world()
        ux, uz = self._grid_to_unity(int(world_xy[0]), int(world_xy[1]))
        return {
            "prior": self._last_prior,
            "audio_like": self._last_audio_like,
            "visual_like": self._last_visual_like,
            "log_visual": self._last_log_visual,
            "log_audio": self._last_log_audio,
            "log_posterior": self._last_posterior,
            "est_world_position": {"x": float(ux), "y": 0.0, "z": float(uz)},
            "observation": self._last_obs,
            "step": self.steps_taken,
            "position": pos,
            "rotation": rot,
            "action": int(self.last_action),
        }

    def get_estimate_snapshot(self) -> dict:
        """Return final estimate (egocentric + world Unity coords if derivable)."""
        est = {
            "step": self.steps_taken,
            "est_theta_deg": float(math.degrees(getattr(self, "est_theta", 0.0))),
            "est_r": float(getattr(self, "est_r", 0.0)),
            "theta_uncertainty": float(getattr(self, "theta_uncertainty", 0.0)),
            "r_uncertainty": float(getattr(self, "r_uncertainty", 0.0)),
        }
        try:
            world_xy = self._estimate_to_world()
            ux, uz = self._grid_to_unity(int(world_xy[0]), int(world_xy[1]))
            est["est_world_position"] = {"x": float(ux), "y": 0.0, "z": float(uz)}
        except Exception:
            pass
        return est

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        
        # Track episode number
        if not hasattr(self, 'current_episode'):
            self.current_episode = 0
        else:
            self.current_episode += 1
            
        self.spawn_objects()

        # Store initial conditions for logging
        self.initial_agent_pos = getattr(self, 'agent_pos', np.array([0, 0])).copy()
        self.initial_agent_heading = getattr(self, 'agent_heading', 0.0)

        # Initialize episode tracking lists
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.last_episode_success = None
        self.last_episode_final_reward = None

        # Reset belief
        self.est_r = float(max(self.grid_w, self.grid_h) // 2)
        self.est_theta = 0.0
        self.theta_uncertainty = math.pi
        self.r_uncertainty = float(max(self.grid_w, self.grid_h))
        self.prev_theta_uncertainty = self.theta_uncertainty
        self.prev_r_uncertainty = self.r_uncertainty

        # Reset all last-state variables to avoid stale state from previous episodes
        self._last_posterior = np.ones((R_GRID.shape[0], THETA_GRID.shape[0])) / (R_GRID.shape[0] * THETA_GRID.shape[0])
        self._last_audio_like = np.ones((R_GRID.shape[0], THETA_GRID.shape[0])) / (R_GRID.shape[0] * THETA_GRID.shape[0])
        self._last_visual_like = np.ones((R_GRID.shape[0], THETA_GRID.shape[0])) / (R_GRID.shape[0] * THETA_GRID.shape[0])
        self._last_heading = self.agent_heading
        self._last_prior = self._last_posterior.copy()
        self._last_log_prior = np.log(self._last_posterior + 1e-12)
        self._last_log_visual = np.log(self._last_visual_like + 1e-12)
        self._last_log_audio = np.zeros_like(self._last_posterior)
        self._last_log_post = np.log(self._last_posterior + 1e-12)
        self.last_agent_pos = self.agent_pos.copy()

        # Reset reward history for new episode
        self.reward_history = []
        self.theta_uncertainty_deltas = []
        self.r_uncertainty_deltas = []
        self.step_costs = []
        self.episode_ends = []
        self.last_action = 0
        
        # Initialize movement counters
        self.head_turns = 0
        self.locomotion_steps = 0

        # Initialize egocentric history buffers
        self.obs_theta_history.clear()
        self.obs_r_history.clear()
        self.action_history.clear()
        self.remaining_steps_history.clear()
        for _ in range(self.history_length):
            self.obs_theta_history.append(self.est_theta)
            self.obs_r_history.append(self.est_r)
            # Prefill with "not an action" (5) as neutral action
            self.action_history.append(5)
            self.remaining_steps_history.append(self.max_steps)

        # Build vehicle_specs by prefab type counts and initialize map via Unity client
        color_to_prefab = {0: "black", 1: "red", 2: "white"}
        vehicles = []

        for i, obj in enumerate(self.objects):
            prefab = color_to_prefab.get(int(obj.get("color", 0)), "black")
            gx, gy = int(obj["pos"][0]), int(obj["pos"][1])
            # Transform the vehicle position from grid to Unity coordinates
            ux, uz = self._grid_to_unity(gx, gy)
            vehicles.append({
                "name": f"{prefab.capitalize()}_Car_{i}",
                "position": {"x": ux, "y": 0.6, "z": uz},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                "isActive": True,
                "isTarget": (i == self.target_idx),
                "prefabType": prefab
            })

        # Agent transform for Unity (convert grid → Unity coordinates)
        ux_a, uz_a = self._grid_to_unity(int(self.agent_pos[0]), int(self.agent_pos[1]))
        agent_position = {"x": ux_a, "y": 0.0, "z": uz_a}
        agent_rotation = {"x": 0.0, "y": float(math.degrees(self.agent_heading)), "z": 0.0}

        if self.use_unity and self.client is not None:
            try:
                self.client.initialize_map(
                    vehicles=vehicles,
                    agent_position=agent_position,
                    agent_rotation=agent_rotation
                )
            except Exception as e:
                print(f"Failed to initialize Unity map with vehicles: {e}")

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # Record state before taking action
        current_state = {
            "agent_pos": self.agent_pos.copy(),
            "agent_heading": self.agent_heading,
            "est_theta": getattr(self, 'est_theta', 0.0),
            "est_r": getattr(self, 'est_r', 0.0),
            "theta_uncertainty": getattr(self, 'theta_uncertainty', math.pi),
            "r_uncertainty": getattr(self, 'r_uncertainty', float(max(self.grid_w, self.grid_h)))
        }
        self.episode_states.append(current_state)
        self.episode_actions.append(action)
        self.last_action = action
        if hasattr(self, 'action_history'):
            self.action_history.append(int(action))

        self.steps_taken += 1
        
        if action == 0:
            # Turn left
            self.head_turns += 1
            delta = -math.radians(self.turn_angle)
            self.agent_heading = self._wrap_angle(self.agent_heading + delta)
            self._rotate_beliefs_by(delta)
            if self.use_unity and self.client is not None:
                self.client.turn_left(self.turn_angle)
        elif action == 1:
            # Turn right
            self.head_turns += 1
            delta = math.radians(self.turn_angle)
            self.agent_heading = self._wrap_angle(self.agent_heading + delta)
            self._rotate_beliefs_by(delta)
            if self.use_unity and self.client is not None:
                self.client.turn_right(self.turn_angle)
        elif action == 2:
            # Move forward in the current direction
            self.locomotion_steps += 1
            new_pos = self.agent_pos + self._heading_vec() * self.pace
            # new_pos = self.agent_pos + self.agent_heading() * self.pace
            new_pos_int = np.array([int(new_pos[0]), int(new_pos[1])])
            
            # Check for collision (out of bounds or object collision)
            collision = False
            collision_reason = ""
            
            # Check bounds
            if (new_pos_int[0] < 0 or new_pos_int[0] >= self.grid_w or 
                new_pos_int[1] < 0 or new_pos_int[1] >= self.grid_h):
                collision = True
                collision_reason = "out_of_bounds"
            
            # Original simpler object collision check (before footprint handling)
            elif any(np.array_equal(new_pos_int, obj["pos"]) for obj in self.objects):
                collision = True
                collision_reason = "object_collision"
            
            if collision:
                # Apply collision penalty and end episode
                collision_reward = -self.collision_penalty
                self.episode_rewards.append(collision_reward)
                self.last_episode_success = False
                self.last_episode_final_reward = collision_reward
                
                # Get observation for final state
                obs = self._get_obs()
                info = self._get_info()
                info.update({
                    "collision": True,
                    "collision_reason": collision_reason,
                    "collision_penalty": self.collision_penalty
                })
                
                # Log episode data on collision
                self.log_episode_data(episode_num=self.current_episode)
                return obs, collision_reward, True, False, info
            else:
                # Move is valid, update position
                self.agent_pos = new_pos
                agent_pos_unity = self._grid_to_unity(self.agent_pos[0], self.agent_pos[1])
                if self.use_unity and self.client is not None:
                    self.client.move_to(agent_pos_unity[0], agent_pos_unity[1])
        elif action == 3:
            if self.use_unity and self.client is not None:
                self.client.commit()  # Simulate commit action. End of episode.
            # Get commit result before logging
            obs, reward, done, truncated, info = self._commit()
            self.episode_rewards.append(reward)
            self.last_episode_success = info.get('success', False)
            self.last_episode_final_reward = reward
            # Log episode data on commit
            self.log_episode_data(episode_num=self.current_episode)
            return obs, reward, done, truncated, info
        elif action == 4:
            # Stay in place, no action taken
            if self.use_unity and self.client is not None:
                self.client.stay()
            
        obs = self._get_obs()
        reward = self._compute_step_reward(action)
        self.episode_rewards.append(reward)
        done = self.steps_taken >= self.max_steps

        # Mark episode end if max steps reached
        if done and hasattr(self, 'reward_history'):
            self.episode_ends.append(len(self.reward_history))
            self.last_episode_success = False  # Episode ended without commit
            self.last_episode_final_reward = reward
            # Log episode data on max steps reached
            self.log_episode_data(episode_num=self.current_episode)

        truncated = False
        info = self._get_info()
        return obs, reward, done, truncated, info


    def _heading_vec(self) -> np.ndarray:
        dx = math.sin(self.agent_heading)
        dy = -math.cos(self.agent_heading)
        # dx = int(round(math.sin(self.agent_heading)))
        # dy = int(round(-math.cos(self.agent_heading)))
        return np.array([dx, dy])

    def _commit(self):
        obs = self._get_obs()
        if self.correct_object:
            success = self._is_commit_correct()
        else:
            success = self._is_commit_correct_footprint()
        base_success = 10.0
        base_fail = -5.0
        reward = base_success if success else base_fail
        done = True
        info = self._get_info()

        _, _, r_true, theta_true = self._geometric_audio_features()
        r_error = abs(r_true - self.est_r)
        theta_error = abs(self._angle_diff(theta_true, self.est_theta))

        # Check if failure is due to closer distractor
        fail_due_to_distractor = False
        if not success:
            est_pos = self._estimate_to_world()
            target_pos = self.target["pos"]
            est_target_dist = np.linalg.norm(est_pos - target_pos)
            for i, obj in enumerate(self.objects):
                if i == self.target_idx:
                    continue
                if np.linalg.norm(est_pos - obj["pos"]) < est_target_dist:
                    fail_due_to_distractor = True
                    break
        info.update({
            "success": success,
            "reward": reward,
            "r_error": r_error,
            "theta_error": theta_error,
            "steps_taken": self.steps_taken,
            "head_turns": self.head_turns,
            "locomotion_steps": self.locomotion_steps,
            "fail_due_to_distractor": fail_due_to_distractor,
        })
        return obs, reward, done, False, info


    def _is_commit_correct(self):
        """
        Adaptive success metric:
        - If the target is within a large acceptable boundary (theta_error < large_theta_bound, r_error < large_r_bound),
          AND no distractor is closer to the estimate than the target within that boundary -> success.
        """
        # Define larger acceptance bounds
        large_theta_bound = self.theta_error_bound * 2.0
        large_r_bound = self.r_error_bound * 2.0

        # Get true target location
        r_true, theta_true = self._get_head_target_angle()
        theta_error = abs(self._angle_diff(theta_true, self.est_theta))
        r_error = abs(r_true - self.est_r)

        # Early reject if target is far outside large boundary
        if theta_error > large_theta_bound or r_error > large_r_bound:
            return False

        # Check if any distractor is closer to estimate than the target (within the large boundary)
        est_pos = self._estimate_to_world()
        target_pos = self.target["pos"]
        est_target_dist = np.linalg.norm(est_pos - target_pos)

        for i, obj in enumerate(self.objects):
            if i == self.target_idx:
                continue
            d = np.linalg.norm(est_pos - obj["pos"])
            if d < est_target_dist:
                return False

        return True

    def _is_commit_correct_footprint(self) -> bool:
        """
        Success if the estimated world position lies within the target car's
        2x4 grid-cell footprint. This uses the same footprint definition as
        `_car_footprint_map()` and avoids any nearest-car checks.

        Returns:
            bool: True iff the estimated (grid) cell belongs to the target's footprint.
        """
        try:
            # Build occupancy map of all car footprints: (gx, gy) -> object index
            occ = self._car_footprint_map()

            # Convert current estimate (r, theta) into a clipped world grid cell
            world_xy = self._estimate_to_world()
            gx, gy = int(world_xy[0]), int(world_xy[1])

            # Check which object (if any) occupies that cell
            hit_idx = occ.get((gx, gy), None)
            return hit_idx == self.target_idx
        except Exception:
            # Be conservative on error
            return False
    

    def _estimate_to_world(self):
        """Convert (est_r, est_theta) relative to agent to world coordinates."""
        # Agent-centered to world conversion
        dx = self.est_r * math.sin(self.agent_heading + self.est_theta)
        dy = -self.est_r * math.cos(self.agent_heading + self.est_theta)
        world = self.agent_pos + np.array([int(round(dx)), int(round(dy))])
        world[0] = int(np.clip(world[0], 0, self.grid_w - 1))
        world[1] = int(np.clip(world[1], 0, self.grid_h - 1))
        return world

    def _compute_step_reward(self, action: int) -> float:
        # -----------------------
        # Calculate reward
        # -----------------------
        
        step_cost = self.step_penalty
        if action in (0, 1):
            step_cost += self.turn_penalty
        elif action == 2:
            step_cost += self.forward_penalty


        # Calculate reward based on change in uncertainty
        prev_theta_uncertainty = getattr(self, "prev_theta_uncertainty", self.theta_uncertainty)
        theta_uncertainty_delta = max(0.0, prev_theta_uncertainty - self.theta_uncertainty)

        prev_r_uncertainty = getattr(self, "prev_r_uncertainty", self.r_uncertainty)
        r_uncertainty_delta = max(0.0, prev_r_uncertainty - self.r_uncertainty)
        
        # reward = (theta_uncertainty_delta + r_uncertainty_delta) - step_cost
        reward = -step_cost

        # Save current uncertainty for next step comparison
        self.prev_theta_uncertainty = self.theta_uncertainty
        self.prev_r_uncertainty = self.r_uncertainty

        return reward
    
    # ---------------------------------------------------------------------
    #  World setup / physics
    # ---------------------------------------------------------------------

    def spawn_objects(self):
        if self.mode == "train":
            self._spawn_train_layout()
        else:
            self._spawn_test_layout()

    def _spawn_train_layout(self):
        """Spawn a randomized training layout using a logical 13x29 grid
        with 16 non-overlapping slots (4 columns x 4 rows). Each car occupies
        an 8-cell footprint (2 cells wide x 4 cells tall). We first pick the
        target slot, then fill remaining cars as distractors in unused slots.

        The environment may use any square `self.grid_size`; we map the
        logical 13x29 grid to the square grid via linear scaling so all
        downstream rendering and logic continue to work.
        """
        rng = np.random.default_rng()

        # Car footprint: 2 (x) by 4 (y) cells -> 8 cells total
        CAR_W, CAR_H = 2, 4

        # Slot anchors (top-left cell of the 2x4 footprint) for a 4x4 layout.
        # Columns are correct; rows start at 0 (top of the grid).
        col_anchors = [1, 4, 7, 10]
        row_anchors = [0, 10, 15, 25]

        # Validate anchors within logical bounds
        def _in_bounds(x0, y0):
            return (
                0 <= x0 < self.grid_w and 0 <= y0 < self.grid_h and
                x0 + CAR_W - 1 < self.grid_w and y0 + CAR_H - 1 < self.grid_h
            )

        slot_topleft = []  # list of (x0, y0) for each slot (row-major)
        for ry in row_anchors:
            for rx in col_anchors:
                if not _in_bounds(rx, ry):
                    raise ValueError(f"Slot anchor ({rx},{ry}) outside 13x29 for 2x4 footprint")
                slot_topleft.append((rx, ry))

        # Helper: footprint cells (logical) covered by a 2x4 car at (x0,y0)
        def footprint_cells(x0: int, y0: int):
            return [(x0 + dx, y0 + dy) for dx in range(CAR_W) for dy in range(CAR_H)]

        # Helper: center of 2x4 footprint (logical coords)
        def footprint_center(x0: int, y0: int):
            # For 2x4 footprint, choose (x0+1, y0+2) as the center cell
            return x0 + CAR_W // 2, y0 + CAR_H // 2

        n_slots_total = len(slot_topleft)  # 16

        # --- Gaussian sampling for counts (truncated) ---
        def _sample_trunc_norm_int(mean: float, std: float, lo: int, hi: int) -> int:
            val = int(round(float(rng.normal(loc=mean, scale=std))))
            return int(np.clip(val, lo, hi))

        # Total number of cars (including target), centered around 5
        n_to_place = _sample_trunc_norm_int(mean=7.0, std=5.0, lo=1, hi=n_slots_total)
        # Number of visual distractors (non-target cars that share target color), centered around 1
        n_visual_distractors = _sample_trunc_norm_int(mean=2.0, std=2.0, lo=0, hi=max(0, n_to_place - 1))

        # Store for logging/consistency
        self.n_objects = n_to_place

        # Reset containers
        self.objects = []

        # Pick target slot first
        target_slot_idx = int(rng.integers(0, n_slots_total))
        used = {target_slot_idx}

        # Place target car
        tx0, ty0 = slot_topleft[target_slot_idx]
        tcx, tcy = footprint_center(tx0, ty0)
        self.objects.append({
            "pos": (tcx, tcy),
            "color": int(rng.integers(0, 3)),  # random prefab/color
        })
        self.target_idx = 0
        self.target = self.objects[self.target_idx]
        target_center_logic = (tcx, tcy)

        # Prepare controlled color assignment for distractors
        target_color = int(self.objects[0]["color"])  # color of the target
        same_color_needed = int(min(n_visual_distractors, max(0, n_to_place - 1)))

        # Place remaining distractors in random distinct slots
        all_indices = list(range(n_slots_total))
        rng.shuffle(all_indices)
        for idx in all_indices:
            if idx in used or len(self.objects) >= n_to_place:
                continue
            used.add(idx)
            x0, y0 = slot_topleft[idx]
            cx, cy = footprint_center(x0, y0)

            # Enforce that exactly `same_color_needed` distractors share the target color
            if same_color_needed > 0:
                color_val = target_color
                same_color_needed -= 1
            else:
                # Choose a color different from the target's
                other_colors = [0, 1, 2]
                try:
                    other_colors.remove(target_color)
                except ValueError:
                    pass
                color_val = int(rng.choice(other_colors)) if other_colors else target_color

            self.objects.append({
                "pos": (cx, cy),
                "color": int(color_val),
            })

        # --- Randomize agent anywhere not intersecting any car footprint ---
        # Build a set of occupied logical cells from all placed cars
        occupied = set()
        for idx in used:
            x0, y0 = slot_topleft[idx]
            for cell in footprint_cells(x0, y0):
                occupied.add(cell)

        # Sample an agent logical cell that is not inside any car
        # footprint and at least 3 cells away from the target center.
        # Note: n_objects and n_visual_distractors are sampled from Gaussians above.
        free_cells = [
            (x, y)
            for x in range(self.grid_w)
            for y in range(self.grid_h)
            if (x, y) not in occupied and ((x - target_center_logic[0]) ** 2 + (y - target_center_logic[1]) ** 2) >= 3 ** 2
        ]
        if not free_cells:
            ax, ay = self.grid_w // 2, self.grid_h // 2
        else:
            ax, ay = free_cells[int(rng.integers(0, len(free_cells)))]

        self.agent_pos = np.array([ax, ay])
        # Generate heading options based on turn_angle
        turn_angle_rad = math.radians(self.turn_angle)
        num_directions = int(2 * math.pi / turn_angle_rad)
        heading_options = [i * turn_angle_rad for i in range(num_directions)]
        self.agent_heading = rng.choice(heading_options)

    def _spawn_test_layout(self):
        """Load test layout from maps/maps.json file."""
        import os
        
        # Load maps from JSON file
        maps_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "maps", "maps.json")
        if not os.path.exists(maps_path):
            print(f"Maps file not found at {maps_path}, falling back to default layout")
            self._spawn_default_test_layout()
            return
            
        try:
            with open(maps_path, 'r') as f:
                maps_data = json.load(f)
        except Exception as e:
            print(f"Failed to load maps from {maps_path}: {e}, falling back to default layout")
            self._spawn_default_test_layout()
            return

        # participant_id = participants[self.current_episode % len(participants)]
        
        participant_maps = maps_data[self.test_map_pid]

        # Select map based on current episode (cycling through available maps for this participant)
        map_data = participant_maps[self.current_episode % len(participant_maps)]
        
        # Clear existing objects
        self.objects = []
        
        # Color mapping from prefabType to our internal color codes
        prefab_to_color = {"black": 0, "red": 1, "white": 2}
        
        # Load vehicles from map data
        vehicles = map_data.get("vehicles", [])
        self.map_id = map_data.get("map_id", "M0001")
        target_idx = None
        
        for i, vehicle in enumerate(vehicles):
            # Convert Unity coordinates to grid coordinates
            unity_x = vehicle["position"]["x"]
            unity_z = vehicle["position"]["z"]
            grid_x, grid_y = self._unity_to_grid(unity_x, unity_z)
            
            # Get color from prefab type
            prefab_type = vehicle.get("prefabType", "black")
            color = prefab_to_color.get(prefab_type, 0)
            
            # Create object
            obj = {
                "pos": np.array([grid_x, grid_y]),
                "color": color
            }
            self.objects.append(obj)
            
            # Track target index
            if vehicle.get("isTarget", False):
                target_idx = i
        
        # Set target information
        if target_idx is not None:
            self.target_idx = target_idx
            self.target = self.objects[self.target_idx]
        else:
            # Fallback: make first object the target
            if self.objects:
                self.target_idx = 0
                self.target = self.objects[0]
            else:
                print("No objects loaded from map, falling back to default layout")
                self._spawn_default_test_layout()
                return
        
        # Load agent position and heading from map data
        agent_data = map_data.get("agent", {})
        if "position" in agent_data:
            agent_unity_x = agent_data["position"]["x"]
            agent_unity_z = agent_data["position"]["z"]
            agent_grid_x, agent_grid_y = self._unity_to_grid(agent_unity_x, agent_unity_z)
            self.agent_pos = np.array([agent_grid_x, agent_grid_y])
        else:
            # Default position
            self.agent_pos = np.array([self.grid_w // 2, self.grid_h - 2])
        
        if "rotation" in agent_data:
            # Convert Unity Y rotation (degrees) to radians
            unity_y_rotation = agent_data["rotation"]["y"]
            # Unity Y rotation: 0=North, 90=East, 180=South, 270=West
            # Our heading: 0=North, π/2=East, π=South, 3π/2=West
            self.agent_heading = math.radians(unity_y_rotation)
        else:
            # Default heading (north)
            self.agent_heading = 0.0
            
        print(f"Loaded map {map_data.get('map_id', 'unknown')} for participant p01 "
              f"with {len(self.objects)} objects, target at index {self.target_idx}")

    def _spawn_default_test_layout(self):
        """Deterministic default test layout as fallback."""
        rng = np.random.default_rng()
        self.objects = []

        # Agent fixed near bottom center, facing north
        self.agent_pos = np.array([self.grid_w // 2, self.grid_h - 2])
        self.agent_heading = 0.0

        # Target at visual center
        tx = self.grid_w // 2
        ty = self.grid_h // 2
        self.target_color = int(rng.integers(0, 3))  # random color
        self.objects.append({
            "pos": np.array([tx, ty]),
            "color": self.target_color
        })
        self.target_idx = 0
        self.target = self.objects[self.target_idx]

        # Two occluders roughly between agent and target (non‑emitting)
        mid1 = self.agent_pos + (np.array([tx, ty]) - self.agent_pos) // 3
        mid2 = self.agent_pos + 2 * (np.array([tx, ty]) - self.agent_pos) // 3
        for mid in (mid1, mid2):
            mid = np.array(mid)
            mid[0] = int(np.clip(mid[0], 0, self.grid_w - 1))
            mid[1] = int(np.clip(mid[1], 0, self.grid_h - 1))
            if not any(np.array_equal(o["pos"], mid) for o in self.objects):
                self.objects.append({
                    "pos": mid.astype(int),
                    "color": int(rng.integers(0, 3)),  # random color
                })

        # Fill remaining distractors at random edges (some emitting at target freq)
        rng = np.random.default_rng(123)
        while len(self.objects) < self.n_objects:
            edge = int(rng.integers(0, 4))
            if edge == 0:
                x, y = int(rng.integers(0, self.grid_w)), 0
            elif edge == 1:
                x, y = int(rng.integers(0, self.grid_w)), self.grid_h - 1
            elif edge == 2:
                x, y = 0, int(rng.integers(0, self.grid_h))
            else:
                x, y = self.grid_w - 1, int(rng.integers(0, self.grid_h))
            pos = np.array([x, y])
            if np.array_equal(pos, self.agent_pos) or any(np.array_equal(o["pos"], pos) for o in self.objects):
                continue
            self.objects.append({
                "pos": pos,
                "color": int(rng.integers(0, 3)),
            })

    def _unity_to_grid(self, ux: float, uz: float,
                       x_min: float = 25.0, x_max: float = 38.0,
                       z_min: float = -14.5, z_max: float = 14.5) -> tuple[int, int]:
        """Map Unity world coordinates (x, z) to grid indices (gx, gy).
        Inverse of _grid_to_unity method.
        """
        # Guard against division by zero for degenerate ranges
        x_range = max(x_max - x_min, 1e-6)
        z_range = max(z_max - z_min, 1e-6)
        gw = max(1, self.grid_w - 1)
        gh = max(1, self.grid_h - 1)
        
        # Normalize to [0,1]
        nx = (ux - x_min) / x_range
        nz = (z_max - uz) / z_range  # note: z decreases with y
        
        # Map to grid coordinates
        gx = int(round(nx * gw))
        gy = int(round(nz * gh))
        
        # Clamp to valid grid bounds
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))
        
        return gx, gy


    def _objects_in_fov(self):
        """
        Return visible objects within FOV as a list of (obj_idx, r, theta), filtering out occluded ones.
        """
        candidates = []
        for i, obj in enumerate(self.objects):
            pos = np.array(obj["pos"])
            dx, dy = pos - self.agent_pos
            theta = (math.atan2(dx, -dy) - self.agent_heading + math.pi) % (2 * math.pi) - math.pi
            if abs(theta) <= self.fov_angle / 2:
                r = math.hypot(dx, dy)
                candidates.append((i, r, theta))

        visible_objs = []
        for i, r, theta in candidates:
            blocked = False
            for j, r2, theta2 in candidates:
                if i == j:
                    continue
                if r2 < r:
                    ang_diff = abs(((theta - theta2 + math.pi) % (2 * math.pi)) - math.pi)
                    if ang_diff < math.radians(5):
                        blocked = True
                        break
            if not blocked:
                visible_objs.append((i, r, theta))

        visible_objs.sort(key=lambda x: x[1])
        return visible_objs
        

    def _car_footprint_map(self):
        """Map each occupied grid cell (gx, gy) -> object index for 2x4 car footprints."""
        CAR_W, CAR_H = 2, 4
        occ = {}
        for idx, obj in enumerate(self.objects):
            cx, cy = int(obj["pos"][0]), int(obj["pos"][1])
            x0 = cx - CAR_W // 2
            y0 = cy - CAR_H // 2
            for dx in range(CAR_W):
                for dy in range(CAR_H):
                    gx = x0 + dx; gy = y0 + dy
                    if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                        occ[(gx, gy)] = idx
        return occ

    def _los_visible_free_mask(self):
        """Boolean (r,theta) mask of LOS-visible FREE-SPACE (up to, not including, first occluder)."""
        thetas, rs = THETA_GRID, R_GRID
        mask = np.zeros((len(rs), len(thetas)), dtype=bool)
        if not len(thetas) or not len(rs):
            return mask
        fov_half = float(self.fov_angle) / 2.0
        # thetas are already head-relative in [-pi, pi]; simple abs check
        theta_idx = np.where(np.abs(thetas) <= (fov_half + 1e-9))[0]
        occ = set(self._car_footprint_map().keys())
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        for j in theta_idx:
            theta_rel = float(thetas[j])
            for i, r in enumerate(rs):
                wx = ax + int(round(r * math.sin(self.agent_heading + theta_rel)))
                wy = ay + int(round(-r * math.cos(self.agent_heading + theta_rel)))
                if not (0 <= wx < self.grid_w and 0 <= wy < self.grid_h):
                    break
                if (wx, wy) in occ:
                    break  # stop at first occluder; do not mark it here (free-space only)
                mask[i, j] = True
        return mask


    def _geometric_audio_features(self) -> Tuple[int, float, float, float, float]:
        """
        Return ITD/ILD observation considering multiple simultaneous sound sources.
        The agent knows the target frequency and filters non-matching sources (ignored in mixture).
        """

        itd_mixture = []
        ild_mixture = []
        weights = []
        
        dx, dy = self.target["pos"] - self.agent_pos
        r_obj = math.hypot(dx, dy)
        theta_obj = (math.atan2(dx, -dy) - self.agent_heading + math.pi) % (2 * math.pi) - math.pi

        # Woodworth ITD model for this source
        itd = (self.head_radius / self.speed_of_sound) * (abs(theta_obj) + math.sin(abs(theta_obj)))
        itd *= np.sign(theta_obj)

        # Shaw ILD model for this source
        ild = 0.18 * math.sqrt(max(1e-6, abs(math.sin(theta_obj))))
        ild *= np.sign(theta_obj)

        # Weight inversely proportional to distance (closer = louder)
        w = 1.0 / max(r_obj, 1e-6)
        itd_mixture.append(itd)
        ild_mixture.append(ild)
        weights.append(w)

        if len(weights) == 0:
            # No target-frequency source audible (return default very noisy estimate)
            itd_obs = np.random.normal(loc=0.0, scale=self.itd_noise_scale)
            ild_obs = np.random.normal(loc=0.0, scale=self.ild_noise_scale)
        else:
            # Normalize weights
            weights = np.array(weights)
            weights /= (np.sum(weights) + 1e-9)
            # Weighted mixture of binaural cues
            itd_obs = float(np.sum(weights * np.array(itd_mixture)))
            ild_obs = float(np.sum(weights * np.array(ild_mixture)))
            # Add Gaussian noise
            itd_obs += np.random.normal(loc=0.0, scale=self.itd_noise_scale)
            ild_obs += np.random.normal(loc=0.0, scale=self.ild_noise_scale)

        r_true, theta_true = self._get_head_target_angle()
        theta_signed = ((theta_true + math.pi) % (2 * math.pi)) - math.pi
        return itd_obs, ild_obs, r_true, theta_signed

    def _footprint_cells_centered(self, cx: int, cy: int):
        xs = [cx - 1, cx]
        ys = [cy - 2, cy - 1, cy, cy + 1]
        for xx in xs:
            if xx < 0 or xx >= self.grid_w:
                continue
            for yy in ys:
                if yy < 0 or yy >= self.grid_h:
                    continue
                yield xx, yy
                
    def _update_estimate(self, itd_obs: float, ild_obs: float, detected_objects: list[Dict]):
        thetas = THETA_GRID  # grid of θ from -pi to pi
        rs = R_GRID
        sigma_itd = self.itd_noise_scale
        sigma_ild = self.ild_noise_scale

        # Forward models for auditory likelihood
        itd_pred = []
        ild_pred = []
        for theta in thetas:
            # Woodworth ITD model (approx, symmetric)
            itd_model = (self.head_radius / self.speed_of_sound) * (abs(theta) + math.sin(abs(theta)))
            itd_model *= np.sign(theta)
            itd_pred.append(itd_model)
            # Shaw ILD model (approx, using current freq=1 assumption)
            ild_model = 0.18 * math.sqrt(max(1e-6, abs(math.sin(theta))))
            ild_model *= np.sign(theta)
            ild_pred.append(ild_model)
        itd_pred = np.array(itd_pred)
        ild_pred = np.array(ild_pred)

        # Gaussian likelihood (non-log)
        audio_like_theta = np.exp(-(itd_obs - itd_pred) ** 2 / (2 * sigma_itd ** 2)) / (
            math.sqrt(2 * math.pi) * sigma_itd
        )
        audio_like = np.tile(audio_like_theta, (len(rs), 1))

        # --- Audio likelihood in log-space ---
        log_audio_like_theta = -(itd_obs - itd_pred) ** 2 / (2 * sigma_itd ** 2) - np.log(math.sqrt(2 * math.pi) * sigma_itd)
        log_audio_like_theta -= np.max(log_audio_like_theta)
        from scipy.ndimage import gaussian_filter1d
        log_audio_like_theta = gaussian_filter1d(
            log_audio_like_theta,
            sigma=1.0 / (thetas[1] - thetas[0]),
            mode="wrap"
        )

        log_audio_like = np.tile(log_audio_like_theta, (len(rs), 1))

        # --- Visual likelihood ---
        if not hasattr(self, "_last_visual_like"):
            self._last_visual_like = np.ones((len(rs), len(thetas))) / (len(rs) * len(thetas))
            self._last_heading = self.agent_heading

        visual_like = self._last_visual_like.copy()

        # Adjust for head rotation since last step
        heading_change = (self.agent_heading - getattr(self, "_last_heading", self.agent_heading) + math.pi) % (2 * math.pi) - math.pi
        if abs(heading_change) > 1e-6:
            theta_res = thetas[1] - thetas[0]
            shift_cells = int(round(-heading_change / theta_res))
            visual_like = np.roll(visual_like, shift_cells, axis=1)


        # --- Aggregate visual updates from all visible objects, now including color and frequency ---

        def wrap_angle(a):
            # Wrap to (-pi, pi]
            return (a + np.pi) % (2 * np.pi) - np.pi

        def von_mises_ll(delta_theta, kappa):
            # log-likelihood up to an additive constant (we ignore normalization everywhere consistently)
            return kappa * np.cos(delta_theta)

        # Precompute angle grid relative to each footprint cell later (we use thetas array you already have)
        sigma_theta = np.deg2rad(self.sigma_theta_deg)
        # Small-angle relation: kappa ~ 1/sigma^2
        kappa = 1.0 / max(sigma_theta**2, 1e-6)

        # Sharp peaks mode: concentrate visual likelihood strictly on observed car cells
        sharp_visual_peaks = (self.sigma_theta_deg <= 1e-3) and (self.sigma_r_min <= 1e-4) and (self.alpha_r <= 1e-6)

        def _accumulate_visual_peaks(update_map: np.ndarray, cx: int, cy: int, weight: float) -> None:
            for xx, yy in self._footprint_cells_centered(cx, cy):
                dx = float(xx) - float(self.agent_pos[0])
                dy = float(yy) - float(self.agent_pos[1])
                r_cell = math.hypot(dx, dy)
                theta_abs = math.atan2(dx, -dy) % (2 * math.pi)
                theta_cell = wrap_angle(theta_abs - self.agent_heading)
                i_r = int(np.argmin(np.abs(rs - r_cell)))
                theta_diffs = (thetas - theta_cell + np.pi) % (2 * np.pi) - np.pi
                i_th = int(np.argmin(np.abs(theta_diffs)))
                update_map[i_r, i_th] += float(weight)

        if self.mode == "test" and len(detected_objects) > 0:
            self.detected_objects = detected_objects
            print(f"Detected {len(detected_objects)} objects in FOV")
            print("Detected objects:", detected_objects)

            # Start from a zeroed update map
            update_map = np.zeros_like(visual_like)

            # Parameters / helpers
            target_color = int(self.target["color"])  # known from spawn
            lambda_color = float(getattr(self, "lambda_color", 0.0))
            visible = 0

            # FOV mask in current head-relative frame
            fov_half = float(getattr(self, 'fov_angle', math.pi)) / 2.0

            CAR_W, CAR_H = 2, 4
            
            # Build likelihood per detected object
            for i_obj, det in enumerate(detected_objects):
                visible += 1

                # World center of the known object (reliable index mapping is assumed)
                cx, cy = self._unity_to_grid(det["position"]["x"], det["position"]["z"])
                if "Black" in det["name"]:
                    obj_color = 0
                elif "White" in det["name"]:
                    obj_color = 2
                else:
                    obj_color = 1
                
                # cx, cy = int(self.objects[i_obj]['pos'][0]), int(self.objects[i_obj]['pos'][1])
                color_weight = float(self.visible_weight) if obj_color == target_color else lambda_color
                if sharp_visual_peaks:
                    _accumulate_visual_peaks(update_map, cx, cy, color_weight)
                else:
                    # Average contributions across the 2x4 footprint to avoid aliasing
                    contrib = np.zeros_like(update_map)
                    n_cells = 0
                    for xx, yy in self._footprint_cells_centered(cx, cy):
                        # Egocentric polar location of the footprint cell
                        dx = xx - self.agent_pos[0]
                        dy = yy - self.agent_pos[1]
                        r_cell = math.hypot(dx, dy)
                        theta_abs = math.atan2(dx, -dy) % (2 * math.pi)
                        theta_cell = wrap_angle(theta_abs - self.agent_heading)
                        # Range and bearing terms (soft kernel)
                        sigma_r = max(self.alpha_r * r_cell, self.sigma_r_min)
                        range_term = np.exp(-((rs - r_cell) ** 2) / (2.0 * sigma_r ** 2))
                        theta_diff = wrap_angle(thetas - theta_cell)
                        bearing_term = np.exp(von_mises_ll(theta_diff, kappa))
                        contrib += np.outer(range_term, bearing_term)
                        n_cells += 1
                    if n_cells > 0:
                        contrib /= float(n_cells)
                    # Weight by color match and add to global map
                    update_map += color_weight * contrib

            if visible > 0:
                update_map /= max(1, visible)
                visual_like = (1 - self.visual_alpha) * visual_like + self.visual_alpha * update_map
                print("Visual likelihood after update:")
                print(visual_like.shape)
                print(update_map.shape)
            
            else:
                visual_like = self._last_visual_like.copy()
            
        if not self.use_unity and len(detected_objects) == 0:
            visible = self._objects_in_fov()

            if visible:
                update_map = np.zeros_like(visual_like)
                target_color = self.target["color"]  # assumed available from spawn
                lambda_color = getattr(self, "lambda_color", 0.0)

                # FOV mask in current head-relative frame
                fov_half = float(getattr(self, 'fov_angle', math.pi)) / 2.0
                CAR_W, CAR_H = 2, 4
                
                for i_obj, r_obj, theta_obj in visible:
                    # Store the known frequency for this visible object
                    theta_idx = np.argmin(np.abs(thetas - theta_obj))

                    cx, cy = int(self.objects[i_obj]['pos'][0]), int(self.objects[i_obj]['pos'][1])
                    
                    obj_color = self.objects[i_obj]["color"]
                    color_weight = self.visible_weight if obj_color == target_color else lambda_color
                    
                    if sharp_visual_peaks:
                        _accumulate_visual_peaks(update_map, cx, cy, color_weight)
                    else:
                        # Average contributions across the 2x4 footprint to avoid aliasing
                        contrib = np.zeros_like(update_map)
                        n_cells = 0
                        for xx, yy in self._footprint_cells_centered(cx, cy):
                            # Egocentric polar location of the footprint cell
                            dx = xx - self.agent_pos[0]
                            dy = yy - self.agent_pos[1]
                            r_cell = math.hypot(dx, dy)
                            theta_abs = math.atan2(dx, -dy) % (2 * math.pi)
                            theta_cell = wrap_angle(theta_abs - self.agent_heading)
                            # Range and bearing terms (soft kernel)
                            sigma_r = max(self.alpha_r * r_cell, self.sigma_r_min)
                            range_term = np.exp(-((rs - r_cell) ** 2) / (2.0 * sigma_r ** 2))
                            theta_diff = wrap_angle(thetas - theta_cell)
                            bearing_term = np.exp(von_mises_ll(theta_diff, kappa))
                            contrib += np.outer(range_term, bearing_term)
                            n_cells += 1
                        if n_cells > 0:
                            contrib /= float(n_cells)
                        
                        # Weight by color match and add to global map
                        update_map += color_weight * contrib
                   
                update_map /= max(1, len(visible))
                visual_like = (1 - self.visual_alpha) * visual_like + self.visual_alpha * update_map
            else:
                # Explicitly skip updates to preserve prior belief strictly when nothing in FOV
                visual_like = self._last_visual_like.copy()

        # --- Bayesian Filtering with Log-Space Prior Integration ---
        prior = getattr(self, "_last_posterior", np.ones_like(log_audio_like) / log_audio_like.size)

        # --- Rotate prior to current head-relative frame ---
        if hasattr(self, "_last_heading"):
            heading_change = (self.agent_heading - self._last_heading + math.pi) % (2 * math.pi) - math.pi
            if abs(heading_change) > 1e-6:
                theta_res = thetas[1] - thetas[0]
                shift_cells = int(round(-heading_change / theta_res))
                prior = np.roll(prior, shift_cells, axis=1)

        # Motion prediction
        if hasattr(self, "last_agent_pos"):
            move_vec = self.agent_pos - self.last_agent_pos
            move_dist = np.linalg.norm(move_vec)
            if move_dist > 0:
                shift_cells = int(round(move_dist / (R_GRID[1] - R_GRID[0])))
                if shift_cells > 0:
                    prior = np.roll(prior, -shift_cells, axis=0)
                    prior[-shift_cells:, :] = np.ones((shift_cells, prior.shape[1])) / (prior.shape[0] * prior.shape[1])

        self._last_prior = prior.copy()

        # --- Visual exclusion/discounting logic ---
        # --- Direct LOS-based visual discounts (applies every step, independent of target visibility) ---
        # 1) Discount LOS-visible FREE-SPACE
        los_free = self._los_visible_free_mask()
        if np.any(los_free):
            visual_like[los_free] *= (1.0 - float(self.visual_exclusion_decay))


        def wrap_diff(a, b):
            d = (a - b + np.pi) % (2*np.pi) - np.pi
            return d


        # 2) Discount LOS-visible surfaces of DIFFERENT-COLOR objects (stop at first occluder per theta)
        thetas, rs = THETA_GRID, R_GRID
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        fov_half = float(self.fov_angle) / 2.0
        theta_idx = np.where(np.abs(((thetas + np.pi) % (2*np.pi) - np.pi)) <= fov_half)[0]
        occ_map = self._car_footprint_map()  # (gx,gy) -> obj_idx
        to_discount = np.zeros_like(visual_like, dtype=bool)
        tgt_color = int(self.target["color"])

        for j in theta_idx:
            theta_rel = float(thetas[j])
            # March outwards until first occluder
            for i, r in enumerate(rs):
                wx = ax + int(round(r * math.sin(self.agent_heading + theta_rel)))
                wy = ay + int(round(-r * math.cos(self.agent_heading + theta_rel)))
                if not (0 <= wx < self.grid_w and 0 <= wy < self.grid_h):
                    break
                if (wx, wy) in occ_map:
                    obj_idx = occ_map[(wx, wy)]
                    obj_color = int(self.objects[obj_idx]["color"])
                    if obj_color != tgt_color:
                        to_discount[i, j] = True
                    # break  # stop at first occluder; do not touch behind it
                # else: free-space already handled by los_free

        if np.any(to_discount):
            visual_like[to_discount] *= (1.0 - float(self.visual_exclusion_decay))

        # Renormalize after discounts
        s = float(np.sum(visual_like))
        if not np.isfinite(s) or s <= 0:
            visual_like = np.ones_like(visual_like) / visual_like.size
        else:
            visual_like /= s

        # Log-space fusion
        log_prior = np.log(prior + 1e-12)
        log_visual = np.log(visual_like + 1e-12)
        alpha = self.alpha
        visual_audio_ratio = self.visual_audio_ratio
        log_post = (1 - alpha) * log_prior + alpha * ((1-visual_audio_ratio) * log_audio_like + visual_audio_ratio * log_visual)
        log_post -= np.max(log_post)

        posterior = np.exp(log_post)
        posterior /= np.sum(posterior) + 1e-9

        # Store log components for rendering/debugging
        self._last_log_prior = log_prior.copy()
        self._last_log_visual = log_visual.copy()
        self._last_log_audio = log_audio_like.copy()
        self._last_log_post = log_post.copy()

        self._last_posterior = posterior.copy()
        self._last_audio_like = audio_like.copy()
        self._last_visual_like = visual_like.copy()
        self._last_heading = self.agent_heading  # update after rotating
        self.last_agent_pos = self.agent_pos.copy()

        # Maximum a posteriori (MAP) estimate
        idx = np.unravel_index(np.argmax(posterior), posterior.shape)
        self.est_r = rs[idx[0]]
        self.est_theta = thetas[idx[1]]

        # Circular mean and uncertainty for θ
        p_theta = posterior.sum(axis=0)
        C = float(np.sum(p_theta * np.cos(thetas)))
        S = float(np.sum(p_theta * np.sin(thetas)))
        mean_theta = math.atan2(S, C)
        self.est_theta = mean_theta
        R = math.hypot(C, S) / (np.sum(p_theta) + 1e-12)
        R = max(min(R, 1.0), 1e-12)
        self.theta_uncertainty = math.sqrt(max(0.0, -2.0 * math.log(R)))

        mean_r = np.sum(rs * np.sum(posterior, axis=1))
        var_r = np.sum((rs - mean_r) ** 2 * np.sum(posterior, axis=1))
        self.r_uncertainty = math.sqrt(var_r)

    def _extract_audio_cues(self) -> Tuple[float, float]:
        """
        Return (itd_us, ild_db).

        - If mode == 'train': use geometric simulator (unchanged).
        - Else: try reading from Unity's recorded WAV via compute_itd_ild_from_wav.
                Falls back to geometric on error or if no file yet.

        With use_unity = false, it will fall back to the geometric mode, same as train mode
        CHI 2026 paper result is based on both train and test in the geometric simulator.
        """
        # 1) Train mode → geometric (keeps your current behavior)
        if str(getattr(self, "mode", "train")).lower() == "train":
            itd_us, ild_db, _, _ = self._geometric_audio_features() 
            return float(itd_us), float(ild_db)

        # 2) Eval/test mode → WAV-based
        
        try:
            wav_path = None
            if self.use_unity and self.client is not None:
                wav_path = self.client.get_audio_file_path()

            if wav_path and os.path.exists(wav_path):
                # Expect: compute_itd_ild_from_wav returns (itd_seconds, ild_db)
                itd_seconds, ild_db = compute_itd_ild_from_wav(wav_path)
                itd_us = float(itd_seconds) * 1e6  # seconds -> microseconds for consistency with forward model
                return float(itd_us), float(ild_db)
            else:
                # File not ready yet (first frame, etc.) → soft fallback
                # print("   [audio] WAV path not found or file missing; falling back to geometric cues this step.")
                pass
        except Exception as e:
            # print(f"   [audio] WAV-based ITD/ILD extraction failed: {e}. Falling back to geometric.")
            pass

        # 3) Fallback → geometric
        itd_us, ild_db, _, _ = self._geometric_audio_features()
        return float(itd_us), float(ild_db)

    def _get_obs(self):
        # Step 1: Get observation (audio) from Unity, if enabled
        if self.use_unity and self.client is not None:
            if not self.client.get_observation():
                print("Failed to get observation, skipping iteration")
                return
            
            # Step 2: Check if audio file exists (optional)
            if self.client.check_audio_file_exists():
                print(f"   Audio file found: {self.client.get_audio_file_path()}")
            else:
                print("   Audio file not found yet")
            
            # Step 3: Process observation and decide action
            print("2. Processing audio observation...")

            # Load binaural RIR from Unity's recorded .wav and extract ITD/ILD
            # wav_path = self.client.get_audio_file_path()
            # itd, ild = compute_itd_ild_from_wav(wav_path)
            itd, ild = self._extract_audio_cues()

            # Get visual observation
            visual_obs = self.client.get_visual_observation()
            if visual_obs is not None and "detectedObjects" in visual_obs:
                self._update_estimate(itd, ild, visual_obs['detectedObjects'])
            else:
                self._update_estimate(itd, ild, [])
                print("visual observation:", visual_obs)
                print("No detected objects in visual observation")
        else:
            # If Unity is disabled, simulate audio/visual features locally
            itd, ild = self._extract_audio_cues()
            self._update_estimate(itd, ild, [])

        # Stack egocentric estimates directly into history
        self.obs_theta_history.append(self.est_theta)
        self.obs_r_history.append(self.est_r)
        self.remaining_steps_history.append(int(self.max_steps - self.steps_taken))

        last_actions = np.array(list(self.action_history), dtype=np.int32) if hasattr(self, 'action_history') else np.zeros((self.history_length,), dtype=np.int32)

        obs = {
            "est_theta": np.array(self.obs_theta_history.copy(), dtype=np.float32),
            "theta_uncertainty": np.array([self.theta_uncertainty], dtype=np.float32),
            "est_r": np.array(self.obs_r_history.copy(), dtype=np.float32),
            "r_uncertainty": np.array([self.r_uncertainty], dtype=np.float32),
            "last_actions": last_actions,
            "remaining_steps": np.array(self.remaining_steps_history.copy(), dtype=np.int32),
        }

        if self.use_posterior_obs and hasattr(self, "_last_posterior"):
            post = self._last_posterior.astype(np.float32)
            # Normalize defensively
            s = float(post.sum())
            if s <= 0 or not np.isfinite(s):
                post = np.ones_like(post, dtype=np.float32)
                post /= float(post.size)
            else:
                post /= s
            obs["posterior"] = post.flatten()
            # Add a simple scalar summary: Shannon entropy
            ent = -float(np.sum(post * (np.log(post + 1e-12))))
            obs["posterior_entropy"] = np.array([ent], dtype=np.float32)

        self._last_obs = obs
        return obs

        
    

    def _get_info(self):
        info = {
            "theta_uncertainty": self.theta_uncertainty,
            "r_uncertainty": getattr(self, "r_uncertainty", None),
        }
        if hasattr(self, "_last_posterior"):
            info["max_itd_likelihood"] = float(np.max(self._last_posterior))
            info["posterior_distribution"] = self._last_posterior.tolist()
        return info


    # ------------------------------------------------------------------
    #  Utilities & rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return (a - b + math.pi) % (2 * math.pi) - math.pi
    
    def _get_head_target_angle(self) -> Tuple[float, float]:
        # 1) world‐coords vector to the target
        dx, dy = self.target["pos"] - self.agent_pos

        # 2) get absolute bearing
        theta_abs = math.atan2(dx, -dy) % (2*math.pi)

        # 3) get relative bearing
        theta = (theta_abs - self.agent_heading) % (2 * math.pi)

        # 4) range
        r_true = math.hypot(dx, dy)
        return r_true, theta


    def render(self):
        if self.render_mode == "human":
            self._render_ascii()
        elif self.render_mode == "rgb_array":
            img = self._render_matplotlib()
            if self.wandb_run is not None:
                self.wandb_run.log({"render": wandb.Image(img)})
            return img
        else:
            raise ValueError("Unsupported mode: " + self.render_mode)

    def _render_ascii(self):
        grid = np.full((self.grid_h, self.grid_w), ".", dtype="U1")
        for i, obj in enumerate(self.objects):
            r, c = obj["pos"]
            grid[r, c] = f"T{i}" if i == self.target_idx else f"O{i}"
        ar, ac = self.agent_pos
        grid[ar, ac] = "A"
        print("\n".join(" ".join(row) for row in grid))


    def _render_matplotlib(self):
        fig, axs = plt.subplots(1, 4, figsize=(16, 5))
        colors = ['black', 'red', 'white']
        thetas = THETA_GRID
        rs = R_GRID

        # Helper to draw the world map on a given axis (matching style across panels)
        def draw_world(ax_world):
            ax_world.set_aspect('equal')
            ax_world.set_title("World", fontsize=10)
            ax_world.set_xlim(-0.5, self.grid_w - 0.5)
            ax_world.set_ylim(-0.5, self.grid_h - 0.5)
            ax_world.set_xticks([])
            ax_world.set_yticks([])
            ax_world.grid(False)
            ax_world.invert_yaxis()
            CAR_W, CAR_H = 2, 4
            for i, obj in enumerate(self.objects):
                ox, oy = obj["pos"]
                facecolor = colors[obj["color"] % 3]
                alpha_val = 0.35
                edgecolor = 'red' if i == self.target_idx else 'black'
                x0 = ox - CAR_W / 2.0
                y0 = oy - CAR_H / 2.0
                rect = patches.Rectangle((x0, y0), CAR_W, CAR_H,
                                         facecolor=facecolor, alpha=alpha_val,
                                         edgecolor=edgecolor, linewidth=1.5)
                ax_world.add_patch(rect)
                label = f"T{i}" if i == self.target_idx else f"O{i}"
                ax_world.text(ox, oy, label, ha='center', va='center', fontsize=8, color='k')

            # Agent icon and heading arrow
            x_a, y_a = self.agent_pos
            ax_world.plot([x_a], [y_a], marker='o', markersize=4, color='#111111')
            dx = math.sin(self.agent_heading)
            dy = -math.cos(self.agent_heading)
            ax_world.arrow(x_a, y_a, dx, dy,
                           head_width=0.35, head_length=0.7,
                           fc='#111111', ec='#111111', linewidth=0.8,
                           length_includes_head=True)

            for i, obj in enumerate(self.detected_objects):
                x_vis, y_vis = self._unity_to_grid(obj["position"]["x"], obj["position"]["z"])
                circ = patches.Circle((x_vis, y_vis), radius=0.3, facecolor='none', edgecolor='yellow', linewidth=1.5)
                ax_world.add_patch(circ)

            # Draw current estimate as a star on the world map
            try:
                r_true, theta_true = self._get_head_target_angle()
                dx = r_true * math.sin(self.agent_heading + theta_true)
                dy = -r_true * math.cos(self.agent_heading + theta_true)
                world = self.agent_pos + np.array([int(round(dx)), int(round(dy))])
                world[0] = int(np.clip(world[0], 0, self.grid_w - 1))
                world[1] = int(np.clip(world[1], 0, self.grid_h - 1))
                ax_world.scatter([world[0]], [world[1]], c='#FFD700', s=120, marker='*',
                                 edgecolors='red', linewidths=1.2, zorder=10, label='TRUE')

                wx, wy = self._estimate_to_world()
                ax_world.scatter([wx], [wy], c='#FFD700', s=120, marker='*',
                                 edgecolors='blue', linewidths=1.2, zorder=10, label='Estimate')
            except Exception:
                pass

        # Helper to overlay an egocentric (r,theta) array onto world map (matching plot_beliefs.py)
        def overlay_belief_on_world(ax_world, arr, cmap='Greens', alpha=0.5, title=None):
            draw_world(ax_world)
            if arr is None:
                if title:
                    ax_world.set_title(title, fontsize=10)
                return
            try:
                # Determine theta/r grids matching the array
                arr = np.array(arr, dtype=float)
                ths = thetas
                rads = rs
                if arr.shape != (len(rs), len(thetas)):
                    ths = np.linspace(-np.pi, np.pi, arr.shape[1])
                    r_min = float(rs.min()) if hasattr(rs, 'min') else 0.5
                    r_max = float(rs.max()) if hasattr(rs, 'max') else 30.0
                    rads = np.linspace(r_min, r_max, arr.shape[0])

                # Agent pose in grid coords
                gx_a = float(self.agent_pos[0])
                gy_a = float(self.agent_pos[1])
                heading = float(self.agent_heading)

                # Mesh in egocentric polar, convert to grid coords
                theta_grid, r_grid = np.meshgrid(ths, rads, indexing='xy')
                gx = gx_a + r_grid * np.sin(heading + theta_grid)
                gy = gy_a - r_grid * np.cos(heading + theta_grid)

                # Normalize values (log → prob-like, else min-max)
                if np.any(np.isfinite(arr)):
                    if np.nanmax(arr) > 0 and np.nanmin(arr) < 0:
                        arr_plot = np.exp(arr - np.nanmax(arr))
                    else:
                        denom = (np.nanmax(arr) - np.nanmin(arr) + 1e-12)
                        arr_plot = (arr - np.nanmin(arr)) / denom
                else:
                    arr_plot = None

                if arr_plot is not None:
                    sc = ax_world.scatter(gx, gy, c=arr_plot, cmap=cmap, s=12, marker='s', alpha=alpha, linewidths=0)
                    mappable = plt.cm.ScalarMappable(cmap=cmap)
                    mappable.set_array(arr_plot)
                    plt.colorbar(mappable, ax=ax_world, fraction=0.046, pad=0.04)
                if title:
                    ax_world.set_title(title, fontsize=10)
            except Exception as e:
                ax_world.text(0.02, 0.02, f"overlay failed\n{e}", transform=ax_world.transAxes,
                              ha='left', va='bottom', fontsize=8,
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # --- 0: World map with estimate star ---
        ax = axs[0]
        draw_world(ax)

        # --- 1: World + log_visual overlay (match plot_beliefs.py) ---
        ax = axs[1]
        log_visual = getattr(self, "_last_log_visual", None)
        overlay_belief_on_world(ax, log_visual, cmap='Blues', alpha=0.5, title="world + log_visual")

        # --- 2: World + log_audio overlay ---
        ax = axs[2]
        log_audio = getattr(self, "_last_log_audio", None)
        overlay_belief_on_world(ax, log_audio, cmap='Oranges', alpha=0.5, title="world + log_audio")

        # --- 3: World + log_posterior overlay ---
        ax = axs[3]
        log_post = getattr(self, "_last_log_post", None)
        overlay_belief_on_world(ax, log_post, cmap='Greens', alpha=0.5, title="world + log_posterior")

        plt.tight_layout()

        # Save the plot to a numpy array with shape (height, width, 3)
        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        
        # Get the buffer and calculate the actual size
        image_buffer = canvas.tostring_argb()
        buffer_size = len(image_buffer)
        expected_size = height * width * 4  # 4 bytes per pixel (ARGB)
        
        if buffer_size != expected_size:
            # Recalculate dimensions based on actual buffer size
            actual_pixels = buffer_size // 4
            # Try to maintain aspect ratio, but prioritize getting valid dimensions
            if actual_pixels != width * height:
                print(f"Warning: Canvas size mismatch. Expected: {width}x{height} ({expected_size} bytes), Got: {buffer_size} bytes")
                # Use the buffer size to determine actual dimensions
                width = int(np.sqrt(actual_pixels * (width / height)))
                height = actual_pixels // width
        
        image_argb = np.frombuffer(image_buffer, dtype=np.uint8)
        image_argb = image_argb.reshape((height, width, 4))
        # Convert ARGB to RGB by dropping the alpha channel and reordering
        image_rgb = image_argb[:, :, 1:4]
        plt.close(fig)
        return image_rgb
    
    def close(self):
        if self.use_unity and self.client is not None:
            self.client.disconnect()
