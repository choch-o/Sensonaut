import math
import json
import wandb
import gymnasium as gym
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
from agents.wandb_callback import WandbEpisodeCallback
import atexit, signal, os
import numpy as np
import random
import pickle

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym  # ensure gym is available for type hints in extractor


def _mlp(sizes, act=nn.ReLU, layernorm=False):
    layers = []
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            if layernorm:
                layers.append(nn.LayerNorm(sizes[i+1]))
            layers.append(act())
    return nn.Sequential(*layers)

class AVSCombinedExtractorCNN(BaseFeaturesExtractor):
    """
    Custom feature extractor for dict observations with a large 'posterior' vector.
    Treats the posterior as a 2D map (R x THETA) and processes it with a small CNN.
    Branches:
      - 'est_theta' (H,)              -> 128
      - 'theta_uncertainty' (1,)      -> 32
      - 'est_r' (H,)                  -> 128
      - 'r_uncertainty' (1,)          -> 32
      - 'last_actions' MultiDiscrete(H, n=6) -> Embedding(6,8) x H -> 64
      - optional 'posterior' (R*THETA,) -> Conv2d -> GAP -> Linear(32 -> 256)
      - optional 'posterior_entropy' (1,) -> 16
    Final feature is the concatenation of all active branch outputs.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 0):
        spaces = observation_space.spaces
        has_posterior = "posterior" in spaces
        has_posterior_entropy = "posterior_entropy" in spaces

        # --- infer sizes from observation space ---
        H = spaces["est_theta"].shape[0]  # history_length
        # MultiDiscrete cardinality (assumes all the same)
        if isinstance(spaces["last_actions"], gym.spaces.MultiDiscrete):
            nvec = spaces["last_actions"].nvec
            action_card = int(nvec[0])
        else:
            raise ValueError("Expected last_actions to be MultiDiscrete.")

        # Compute the true features_dim up front so SB3 builds heads correctly
        out_dim = 128 + 32 + 128 + 32 + 64  # theta, theta_unc, r, r_unc, last_actions
        if has_posterior:
            out_dim += 256
        if has_posterior_entropy:
            out_dim += 16

        super().__init__(observation_space, features_dim=out_dim)

        # Save attributes used later
        self.has_posterior = has_posterior
        self.has_posterior_entropy = has_posterior_entropy
        self.action_card = action_card
        self.history_len = H

        # --- small MLP branches for low-dim vectors ---
        self.theta_net = _mlp([H, 128, 128], layernorm=True)
        self.theta_unc_net = _mlp([1, 32, 32], layernorm=False)
        self.r_net = _mlp([H, 128, 128], layernorm=True)
        self.r_unc_net = _mlp([1, 32, 32], layernorm=False)

        # --- embedding + projection for last_actions tokens ---
        self.action_emb = nn.Embedding(self.action_card, 8)
        self.action_proj = _mlp([H * 8, 64, 64], layernorm=False)

        # --- posterior branch: process as 2D (R x THETA) with a tiny CNN ---
        if self.has_posterior:
            flat = spaces["posterior"].shape[0]
            # Heuristically recover (R, THETA) by factoring:
            # We know in this project: R=15, THETA=361 (from config/utils).
            # Fall back to (R=15, THETA=flat//15) if 361 not exact.
            R = 30
            TH = flat // R if flat % R == 0 else flat
            self.posterior_shape = (1, R, TH)  # (C=1, H=R, W=TH)

            self.post_cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))  # -> [B, 32, 1, 1]
            )
            self.post_head = nn.Sequential(
                nn.Flatten(),               # -> [B, 32]
                nn.LayerNorm(32),
                nn.Linear(32, 256),
                nn.ReLU()
            )

        if self.has_posterior_entropy:
            self.post_ent_net = _mlp([1, 16, 16], layernorm=False)

    def forward(self, obs: dict) -> th.Tensor:
        # cast to tensors
        est_theta = th.as_tensor(obs["est_theta"], dtype=th.float32)
        theta_unc = th.as_tensor(obs["theta_uncertainty"], dtype=th.float32)
        est_r     = th.as_tensor(obs["est_r"], dtype=th.float32)
        r_unc     = th.as_tensor(obs["r_uncertainty"], dtype=th.float32)
        # last_actions may be indices or a (flattened) one-hot depending on SB3 preprocessing

        t_feat  = self.theta_net(est_theta)
        tu_feat = self.theta_unc_net(theta_unc)
        r_feat  = self.r_net(est_r)
        ru_feat = self.r_unc_net(r_unc)

        # Support either index tensor [B,H], one-hot [B,H,C], or flattened one-hot [B,H*C]
        last_act_raw = th.as_tensor(obs["last_actions"])  # don't force dtype yet

        if last_act_raw.dim() == 3 and last_act_raw.shape[-1] == self.action_card:
            # One-hot per token -> indices
            last_idx = last_act_raw.argmax(-1).long()
        elif last_act_raw.dim() == 2 and last_act_raw.shape[1] == self.history_len * self.action_card:
            # Flattened one-hot -> reshape and argmax
            B = last_act_raw.shape[0]
            last_idx = last_act_raw.view(B, self.history_len, self.action_card).argmax(-1).long()
        elif last_act_raw.dim() == 2 and last_act_raw.shape[1] == self.history_len:
            # Already indices of shape [B, H]
            last_idx = last_act_raw.long()
        else:
            # Fallback: attempt to coerce first H positions
            last_idx = last_act_raw.long()
            if last_idx.dim() > 2:
                last_idx = last_idx.view(last_idx.shape[0], -1)[:, : self.history_len]

        a_emb  = self.action_emb(last_idx)            # [B, H, 8]
        a_flat = a_emb.reshape(a_emb.shape[0], -1)   # [B, H*8] = [B, 32] for H=4
        a_feat = self.action_proj(a_flat)

        feats = [t_feat, tu_feat, r_feat, ru_feat, a_feat]

        if self.has_posterior:
            posterior = th.as_tensor(obs["posterior"], dtype=th.float32)
            B = posterior.shape[0]
            C, R, TH = self.posterior_shape
            # reshape flat -> (B, 1, R, TH), safe with reshape
            p_img = posterior.reshape(B, C, R, TH)
            p_feat = self.post_cnn(p_img)
            p_feat = self.post_head(p_feat)  # -> [B, 256]
            feats.append(p_feat)

        if self.has_posterior_entropy:
            p_ent = th.as_tensor(obs["posterior_entropy"], dtype=th.float32)
            pe_feat = self.post_ent_net(p_ent)
            feats.append(pe_feat)

        return th.cat(feats, dim=1)


SEED = 42

def train_agent(config):
    # Build a descriptive W&B run name
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    alg = str(config.get("model", {}).get("algorithm", "ALG"))
    pol = str(config.get("model", {}).get("policy", "Policy"))
    suffix = config.get("logging", {}).get("name_suffix")
    run_name = f"train-{alg}-{pol}-{suffix}-{ts}" if suffix else f"train-{alg}-{pol}-{ts}"

    run = wandb.init(
        project=config["logging"]["project_name_train"],
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        tags=["train"] + (["grid-search"] if suffix else []),
        reinit=True,
    )

    # if config["environment"]["use_unity"]:
    from envs.unityproxy_env import UnityProxyEnv as Environment
    
    env = Environment(
        render_mode=config["environment"]["render_mode"],
        grid_w=config["environment"]["grid_w"],
        grid_h=config["environment"]["grid_h"],
        n_objects=config["environment"]["n_objects"],
        max_steps=config["environment"]["max_steps"],
        theta_error_bound=config["environment"]["theta_error_bound"],
        r_error_bound=config["environment"]["r_error_bound"],
        step_penalty=config["environment"].get("step_penalty", 0.0),
        forward_penalty=config["environment"]["forward_penalty"],
        turn_penalty=config["environment"]["turn_penalty"],
        collision_penalty=config["environment"]["collision_penalty"],
        fov_angle=math.radians(config["environment"]["fov_angle_deg"]),
        alpha=config["environment"]["alpha"],
        visual_alpha=config["environment"]["visual_alpha"],
        lambda_color=config["environment"]["lambda_color"],
        visual_exclusion_decay=config["environment"]["visual_exclusion_decay"],
        mode=config["environment"]["mode"],
        head_radius=config["physics"]["head_radius"],
        speed_of_sound=config["physics"]["speed_of_sound"],
        itd_noise_scale=config["noise"]["itd_noise_scale"],
        ild_noise_scale=config["noise"]["ild_noise_scale"],
        visual_audio_ratio=config["environment"].get("visual_audio_ratio", 0.7),
        # visual_noise_scale=config["noise"]["visual_noise_scale"],
        sigma_r_min=config["noise"].get("sigma_r_min", 0.25),
        alpha_r=config["noise"].get("alpha_r", 0.12),
        sigma_theta_deg=config["noise"].get("sigma_theta_deg", 15.0),
        turn_angle=config["environment"]["turn_angle"],
        pace=config["environment"]["pace"],
        visible_weight=config["environment"]["visible_weight"],
        history_length=config["environment"]["history_length"],
        debug_polar=config["environment"]["debug_polar"],
        use_unity=config["environment"]["use_unity"],
        wandb_run=run,
        use_posterior_obs=config["environment"]["use_posterior_obs"],
        correct_object=config["environment"].get("correct_object", True),
        time_bonus_weight=config["environment"].get("time_bonus_weight", 0.0),
        time_bonus_success_only=config["environment"].get("time_bonus_success_only", True),
    )
    if config["environment"]["render_mode"] == "rgb_array":
        env = gym.wrappers.RecordVideo(env, f"{config['logging']['video_dir']}/{config['model']['policy']}_steps_{config['model']['total_timesteps']}")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if config["model"]["pretrained_model"]:
        model = PPO.load(config["model"]["pretrained_model"], env=env)
    else:
        policy_kwargs = dict(
            features_extractor_class=AVSCombinedExtractorCNN,
            net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
            # activation_fn=nn.ReLU,
            # ortho_init=False,
        )

        model = PPO(
            policy=config["model"]["policy"],
            env=env,
            policy_kwargs=policy_kwargs,
            device="cuda",              # if available
            verbose=1,
            tensorboard_log=config["logging"]["tensorboard_log_dir"],
            batch_size=4096,
            n_steps=4096,
            learning_rate=3e-4,
        )

        # print(model.policy)  # sanity-check the built networks
        # model = PPO(
        #     config["model"]["policy"],
        #     env,
        #     verbose=1,
        #     tensorboard_log=config["logging"]["tensorboard_log_dir"]
        # )
        
    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_dir = config.get("logging", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_freq = int(config.get("logging", {}).get("checkpoint_freq", 100_000))
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir + "/",
        name_prefix="ppo_",
    )

    def safe_save(path_prefix=os.path.join(checkpoint_dir, "ppo_last_interrupt")):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = f"{path_prefix}_{timestamp}"
        model.save(path)

    atexit.register(safe_save)
    signal.signal(signal.SIGINT,  lambda s,f: (safe_save(), exit(130)))
    signal.signal(signal.SIGTERM, lambda s,f: (safe_save(), exit(143)))

    model.learn(
        total_timesteps=config["model"]["total_timesteps"],
        callback=[WandbCallback(), WandbEpisodeCallback(), checkpoint_cb]
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name_suffix = config.get("logging", {}).get("name_suffix")
    base_name = f"trained_model_{timestamp}"
    if name_suffix:
        base_name = f"{base_name}_{name_suffix}"
    model.save(base_name)
    wandb.finish()


def test_agent(config):
    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def _write_json(path: str, payload: dict):
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[test_agent] Failed to write {path}: {e}")

    def _agent_unity_pose(env):
        try:
            ux, uz = env._grid_to_unity(int(env.agent_pos[0]), int(env.agent_pos[1]))
            return (
                {"x": float(ux), "y": 0.0, "z": float(uz)},
                {"x": 0.0, "y": float(math.degrees(getattr(env, "agent_heading", 0.0))), "z": 0.0}
            )
        except Exception as _:
            return (
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 0.0, "z": 0.0}
            )

    test_modes = config["testing"]["test_modes"]
    model = PPO.load(config["model"]["pretrained_model"])

    run_counter = 1  # incremented per test_mode run
    root_log_dir = config["logging"].get("test_log_dir", "test_logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_log_dir = os.path.join(root_log_dir, f"test_{timestamp}")

    # rng = random.Random(SEED)
    rng = np.random.default_rng()
    from envs.unityproxy_env import UnityProxyEnv as Environment
    for test_mode in test_modes:
        # Build a descriptive W&B run name for tests
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        pol = str(config.get("model", {}).get("policy", "Policy"))
        suffix = config.get("logging", {}).get("name_suffix")
        run_name = f"test-{test_mode}-{pol}-{suffix}-{ts}" if suffix else f"test-{test_mode}-{pol}-{ts}"

        run_number = run_counter
        run_dir = os.path.join(root_log_dir, f"agent_{run_number}")
        _ensure_dir(run_dir)

        PID_FMT = "p{:02d}"
        test_map_pid = PID_FMT.format(int(rng.integers(1, 13)))

        env = Environment(
            render_mode=config["environment"]["render_mode"],
            grid_w=config["environment"]["grid_w"],
            grid_h=config["environment"]["grid_h"],
            n_objects=config["environment"]["n_objects"],
            max_steps=config["environment"]["max_steps"],
            theta_error_bound=config["environment"]["theta_error_bound"],
            r_error_bound=config["environment"]["r_error_bound"],
            step_penalty=config["environment"].get("step_penalty", 0.0),
            forward_penalty=config["environment"]["forward_penalty"],
            turn_penalty=config["environment"]["turn_penalty"],
            collision_penalty=config["environment"]["collision_penalty"],
            fov_angle=math.radians(config["environment"]["fov_angle_deg"]),
            alpha=config["environment"]["alpha"],
            visual_alpha=config["environment"]["visual_alpha"],
            lambda_color=config["environment"]["lambda_color"],
            visual_exclusion_decay=config["environment"]["visual_exclusion_decay"],
            mode=config["environment"]["mode"],
            head_radius=config["physics"]["head_radius"],
            speed_of_sound=config["physics"]["speed_of_sound"],
            itd_noise_scale=config["noise"]["itd_noise_scale"],
            ild_noise_scale=config["noise"]["ild_noise_scale"],
            visual_audio_ratio=config["environment"].get("visual_audio_ratio", 0.7),
            # visual_noise_scale=config["noise"]["visual_noise_scale"],
            sigma_r_min=config["noise"].get("sigma_r_min", 0.25),
            alpha_r=config["noise"].get("alpha_r", 0.12),
            sigma_theta_deg=config["noise"].get("sigma_theta_deg", 15.0),
            turn_angle=config["environment"]["turn_angle"],
            pace=config["environment"]["pace"],
            visible_weight=config["environment"]["visible_weight"],
            history_length=config["environment"]["history_length"],
            debug_polar=config["environment"]["debug_polar"],
            use_unity=config["environment"]["use_unity"],
            # wandb_run=run,
            use_posterior_obs=config["environment"]["use_posterior_obs"],
            correct_object=config["environment"].get("correct_object", True),
            time_bonus_weight=config["environment"].get("time_bonus_weight", 0.0),
            time_bonus_success_only=config["environment"].get("time_bonus_success_only", True),
        )
        env = gym.wrappers.RecordVideo(env, f"{config['logging']['video_dir']}/{config['model']['policy']}_{test_mode}")
        env = gym.wrappers.RecordEpisodeStatistics(env)

        episode_number = 0
        step_number = 0

        obs, info = env.reset()
        obs, info = env.reset()
        # lstm_states = None
        episode_starts = np.array([True])
        episode_dir = os.path.join(run_dir, str(episode_number))
        _ensure_dir(episode_dir)

        # Unwrap until you get to UnityProxyEnv
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        # Scene data (one per episode)
        scene_data = base_env.get_scene_snapshot()
        scene_fname = f"scene_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        _write_json(os.path.join(episode_dir, scene_fname), scene_data)

        for _ in range(50000):  # large cap; episode ends via done/truncated/commit
            # Per-step action
            action, _ = model.predict(
                obs,
                episode_start=episode_starts,
                deterministic=config["model"]["deterministic_test"]
            )
            obs, rewards, done, truncated, info = env.step(action)
            episode_starts = np.array([done or truncated])

            # Step log (one file per step)
            step_data = base_env.get_step_snapshot()
            step_path = os.path.join(episode_dir, f"agent_{step_number}.pkl")
            # _write_json(step_path, step_data)
            with open(step_path, "wb") as f:
                pickle.dump(step_data, f)
            step_number += 1

            if done or truncated:
                # Final estimate (one per episode)
                estimate = base_env.get_estimate_snapshot()
                est_fname = f"estimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                _write_json(os.path.join(episode_dir, est_fname), estimate)

                print(f"Episode {episode_number} ended with success: {info.get('success', False)}")
                # Prepare for next episode
                obs, _ = env.reset()
                episode_starts = np.array([True])
                episode_number += 1
                episode_number = episode_number % 270
                step_number = 0
                episode_dir = os.path.join(run_dir, str(episode_number))
                _ensure_dir(episode_dir)
                scene_data = base_env.get_scene_snapshot()
                scene_fname = f"scene_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                _write_json(os.path.join(episode_dir, scene_fname), scene_data)

                if episode_number % 270 == 0:
                    print("Covered all episodes")
                    run_counter += 1

                    run_number = run_counter
                    run_dir = os.path.join(root_log_dir, f"agent_{run_number}")
                    _ensure_dir(run_dir)
                    test_map_pid = PID_FMT.format(int(rng.integers(1, 13)))


        run_counter += 1


if __name__ == "__main__":
    pass
