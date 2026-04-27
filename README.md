# Sensonaut: Simulating Human Audiovisual Search Behavior

This repository includes a research platform for studying audio-visual search behavior in both humans and reinforcement learning agents. The project combines Python implementation of PPO-based RL agents that learn to locate sound-emitting targets using binaural audio and visual cues along with a Unity-based simulator.

<p align="center">
  <img src="media/teaser.gif" width="600">
</p>

## Overview

The platform consists of three main components:

1. **RL Training & Testing**: Train and test PPO agents to locate a target audiovisual source (e.g., vehicle) in a grid environment (e.g., parking garage) using audio (ITD) and visual observations
2. **Unity Simulation**: A 3D environment for realistic audio-visual rendering, supporting both agent control and human participant studies
3. **Analysis Tools**: Comprehensive tools for comparing human and model behavior, including trajectory analysis, belief visualization, and statistical metrics

## Project Structure

```
Sensonaut/
├── python/                               # Python codebase (RL, analysis, utilities)
│   ├── agent.py                          # Main entry point for training/testing
│   ├── agents/                           # RL agent implementations
│   │   ├── ppo_agent.py                  # PPO training and testing
│   │   └── wandb_callback.py             # Weights & Biases logging
│   ├── envs/                             # Gym environments
│   │   └── unityproxy_env.py             # Main environment (grid world + Unity)
│   ├── analysis/                         # Human vs. model comparison tools
│   │   ├── compare_traj.py               # Trajectory comparison
│   │   ├── belief_action_comparison.py   # Belief and action analysis
│   │   ├── human_actions_and_beliefs.py  # Human behavior inference
│   │   └── plot_beliefs.py               # Belief visualization
│   ├── utils/                            # Shared utilities
│   │   ├── constants.py                  # Centralized constants
│   │   ├── coordinates.py                # Coordinate transformations
│   │   ├── unity_client.py               # Unity communication
│   │   ├── audio_features.py             # ITD computation
│   │   └── metrics.py                    # Evaluation metrics
│   ├── maps/                             # Map generation and visualization
│   ├── scripts/                          # Helper scripts
│   └── configs/                          # Configuration files
│       ├── config_train.yaml             # Config file for training a new model
│       └── config_test.yaml              # Config file for testing a trained model
├── unity/                                # Unity project
│   ├── Assets/                           # Unity assets and scripts
│   └── README.md                         # Unity-specific documentation
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Unity 2022.3 LTS or later
- CUDA-capable GPU (recommended for training)

### Python Setup

```bash
# Clone the repository
git clone https://github.com/choch-o/sensonaut.git
cd sensonaut

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Unity Setup

1. Open the Unity project in `unity/` folder
2. Install required Unity packages (Steam Audio, Meta XR Audio SDK)
3. Configure the scene with appropriate prefabs
4. See `unity/README.md` for detailed Unity setup instructions

## Quick Start

### Training an Agent

```bash
cd python

# Train with default configuration
# Edit config to set pretrained_model path for curriculum learning
python agent.py --config configs/config_train.yaml
```

### Testing a Trained Model

```bash
# Edit config to set pretrained_model path
python agent.py --config configs/config_test.yaml
```

### Grid Search for Hyperparameters

```bash
# Run grid search over parameters defined in config
python agent.py --config configs/config_unity.yaml --grid

# Dry run to see planned jobs
python agent.py --config configs/config_unity.yaml --grid --dry-run
```


## The Task

The agent starts in a parking garage environment with multiple vehicles. One vehicle (the target) emits a sound. The agent must:

1. **Listen**: Use binaural audio cues (ITD) to estimate target direction
2. **Look**: Use visual observations to identify and localize vehicles
3. **Move**: Navigate the environment (turn left/right, move forward)
4. **Commit**: When confident, commit to a target location estimate

### Observation Space

- `est_theta`: Estimated target angle (θ azimuth angle relative to current position)
- `est_r`: Estimated target distance (r radius relative to current position)
- `theta_uncertainty`: Current theta uncertainty
- `r_uncertainty`: Current r uncertainty
- `last_actions`: Recent action history
- `posterior`: Full belief distribution over (r, θ)

### Action Space

- `0`: Turn left
- `1`: Turn right
- `2`: Move forward
- `3`: Commit (end episode)
- `4`: Stay (no movement, collect more evidence)

## Analysis

Refer to `run_analysis.sh` for how to run scripts for analysis. 


## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{cho2026simulating,
  title={Simulating Human Audiovisual Search Behavior},
  author={Cho, Hyunsung and Luo, Xuejing and Lee, Byungjoo and Lindlbauer, David and Oulasvirta, Antti},
  booktitle={Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems},
  pages={1--17},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
