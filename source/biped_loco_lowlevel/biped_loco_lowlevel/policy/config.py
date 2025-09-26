# config.py
# Loads YAML config for robot and policy

import argparse
from omegaconf import DictConfig, OmegaConf


class Cfg(DictConfig):
    policy_checkpoint_path: str
    num_joints: int
    num_observations: int
    num_actions: int
    action_scale: float
    action_limit_lower: float
    action_limit_upper: float
    default_joint_positions: list
    command_velocity: list
    history_length: int

    @staticmethod
    def from_arguments() -> DictConfig:
        parser = argparse.ArgumentParser(description="Biped RL Runner")
        parser.add_argument(
            "--config",
            type=str,
            default="./configs/policy_latest.yaml",
            help="Path to configuration file",
        )
        args = parser.parse_args()

        print("Loading config file from ", args.config)
        cfg = OmegaConf.load(args.config)
        return cfg
