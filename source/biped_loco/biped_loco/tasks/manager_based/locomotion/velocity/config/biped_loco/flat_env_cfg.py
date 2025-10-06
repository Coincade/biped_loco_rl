# Copyright (c) 2024, Biped Loco Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import BipedLocoRoughEnvCfg


@configclass
class BipedLocoFlatEnvCfg(BipedLocoRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards - Adjusted for better locomotion
        self.rewards.track_lin_vel_xy_exp.weight = 2.0  # Increased from 1.0 to encourage forward movement
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.001  # Reduced from -0.005 to allow more movement
        self.rewards.dof_acc_l2.weight = -1e-8  # Reduced from -1.0e-7
        self.rewards.feet_air_time.weight = 1.5  # Increased from 0.75 to encourage stepping
        self.rewards.feet_air_time.params["threshold"] = 0.3  # Lowered threshold for easier achievement
        self.rewards.dof_torques_l2.weight = -1e-7  # Reduced from -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_.*"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class BipedLocoFlatEnvCfg_PLAY(BipedLocoFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
