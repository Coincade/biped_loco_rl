# Biped Loco IsaacLab Task

This directory contains the IsaacLab task implementation for the Biped Loco robot, providing locomotion capabilities using reinforcement learning.

## Overview

The Biped Loco task is a manager-based RL environment that enables training locomotion policies for a bipedal robot. It includes:

- **Flat terrain locomotion**: Basic walking on flat surfaces
- **Rough terrain locomotion**: Advanced walking on uneven terrain
- **Velocity tracking**: Following velocity commands
- **RSL-RL integration**: Compatible with RSL-RL training framework
- **SKRL integration**: Compatible with SKRL training framework

## Directory Structure

```
source/biped_loco/
├── biped_loco/
│   └── tasks/
│       └── manager_based/
│           └── locomotion/
│               └── velocity/
│                   ├── config/
│                   │   └── biped_loco/
│                   │       ├── __init__.py
│                   │       ├── flat_env_cfg.py
│                   │       ├── rough_env_cfg.py
│                   │       ├── robot_cfg.py
│                   │       └── agents/
│                   │           ├── __init__.py
│                   │           ├── rsl_rl_ppo_cfg.py
│                   │           └── skrl_flat_ppo_cfg.yaml
│                   ├── mdp/
│                   │   ├── __init__.py
│                   │   ├── commands.py
│                   │   ├── curriculum.py
│                   │   ├── events.py
│                   │   ├── observations.py
│                   │   └── rewards.py
│                   ├── __init__.py
│                   └── velocity_env_cfg.py
├── scripts/
│   ├── train_biped_loco.py
│   └── play_biped_loco.py
└── README_BIPED_LOCO_TASK.md
```

## Available Environments

### Flat Terrain
- **Isaac-Velocity-Flat-BipedLoco-v0**: Training environment for flat terrain
- **Isaac-Velocity-Flat-BipedLoco-Play-v0**: Play environment for flat terrain

### Rough Terrain
- **Isaac-Velocity-Rough-BipedLoco-v0**: Training environment for rough terrain
- **Isaac-Velocity-Rough-BipedLoco-Play-v0**: Play environment for rough terrain

## Robot Configuration

The Biped Loco robot is configured with:
- **10 joints**: 5 per leg (left_hip_pitch_joint, left_hip_roll_joint, left_knee_roll_joint, left_ankle_roll_joint, left_foot_joint, right_hip_pitch_joint, right_hip_roll_joint, right_knee_roll_joint, right_ankle_roll_joint, right_foot_joint)
- **Joint limits**: Position, velocity, and effort limits
- **Default positions**: Neutral standing pose matching policy_latest.yaml
- **Mass properties**: Realistic mass distribution

## Training

### Using RSL-RL

```bash
# Train on flat terrain
python source/biped_loco/scripts/train_biped_loco.py --task Isaac-Velocity-Flat-BipedLoco-v0

# Train on rough terrain
python source/biped_loco/scripts/train_biped_loco.py --task Isaac-Velocity-Rough-BipedLoco-v0
```

### Using SKRL

```bash
# Train using SKRL
python -m skrl.train --config source/biped_loco/biped_loco/tasks/manager_based/locomotion/velocity/config/biped_loco/agents/skrl_flat_ppo_cfg.yaml
```

## Playing Trained Policies

```bash
# Play trained policy
python source/biped_loco/scripts/play_biped_loco.py --checkpoint /path/to/checkpoint.pt

# Play with specific number of environments
python source/biped_loco/scripts/play_biped_loco.py --num_envs 100 --headless
```

## Configuration

### Environment Configuration

The environment can be configured through the following files:
- `flat_env_cfg.py`: Flat terrain configuration
- `rough_env_cfg.py`: Rough terrain configuration
- `robot_cfg.py`: Robot-specific configuration

### Training Configuration

Training parameters can be adjusted in:
- `rsl_rl_ppo_cfg.py`: RSL-RL PPO configuration
- `skrl_flat_ppo_cfg.yaml`: SKRL PPO configuration

### MDP Components

The MDP (Markov Decision Process) components are defined in:
- `rewards.py`: Reward functions
- `observations.py`: Observation functions
- `commands.py`: Command functions
- `events.py`: Event functions
- `curriculum.py`: Curriculum functions

## Key Features

### Rewards
- **Velocity tracking**: Rewards for following velocity commands
- **Stability**: Penalties for undesired contacts and orientation
- **Efficiency**: Penalties for high action rates and torques
- **Gait quality**: Rewards for proper foot air time

### Observations
- **Base velocity**: Linear and angular velocity of the robot
- **Gravity projection**: Orientation relative to gravity
- **Joint states**: Position and velocity of all joints
- **Commands**: Current velocity commands
- **Height scan**: Terrain height information (rough terrain only)

### Commands
- **Linear velocity**: Forward/backward and side-to-side movement
- **Angular velocity**: Yaw rotation commands

### Events
- **Randomization**: Joint position and base pose randomization
- **Disturbances**: External forces and torques
- **Mass variation**: Random mass changes

## Usage Examples

### Basic Training

```python
from isaaclab.envs import ManagerBasedRLEnv

# Create environment
env = ManagerBasedRLEnv(cfg="Isaac-Velocity-Flat-BipedLoco-v0")

# Get observations
obs = env.reset()
print(f"Observation shape: {obs['policy'].shape}")

# Take random actions
for _ in range(100):
    actions = env.action_space.sample()
    obs, rewards, dones, info = env.step(actions)

env.close()
```

### Custom Configuration

```python
from biped_loco.tasks.manager_based.locomotion.velocity.config.biped_loco.flat_env_cfg import BipedLocoFlatEnvCfg

# Create custom configuration
cfg = BipedLocoFlatEnvCfg()
cfg.scene.num_envs = 1000
cfg.rewards.track_lin_vel_xy_exp.weight = 2.0

# Create environment with custom config
env = ManagerBasedRLEnv(cfg=cfg)
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the biped_loco package is properly installed
2. **CUDA errors**: Use `--cpu` flag for CPU-only training
3. **Memory issues**: Reduce `num_envs` or use `--headless` mode
4. **Checkpoint loading**: Ensure checkpoint path is correct and compatible

### Performance Tips

1. **Use headless mode** for training: `--headless`
2. **Adjust number of environments** based on available memory
3. **Use curriculum learning** for better performance on rough terrain
4. **Monitor training progress** through IsaacLab's logging system

## Contributing

When contributing to this task:
1. Follow the existing code structure
2. Add proper type hints and docstrings
3. Update this README if adding new features
4. Test both flat and rough terrain configurations
5. Ensure compatibility with both RSL-RL and SKRL

## License

This project is licensed under the BSD-3-Clause License. See the LICENSE file for details.
