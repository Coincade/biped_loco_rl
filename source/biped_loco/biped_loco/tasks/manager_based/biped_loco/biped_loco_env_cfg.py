# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##


from .biped_loco import BIPED_LOCO_JOINTS, BIPED_LOCO_CFG # isort:skip


##
# Scene definition
##


@configclass
class BipedLocoSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = BIPED_LOCO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),  # Moderate forward/backward movement for walking
            lin_vel_y=(-0.15, 0.15),  # Moderate side-to-side movement
            ang_vel_z=(-0.8, 0.8),  # Moderate rotation
            heading=(-math.pi, math.pi),
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=BIPED_LOCO_JOINTS,
        scale=0.1,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BIPED_LOCO_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BIPED_LOCO_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # === Startup behaviors ===
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
        mode="startup",
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
        mode="startup",
    )
    add_all_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
        mode="startup",
    )
    scale_all_actuator_torque_constant = EventTerm(
        func=mdp.randomize_actuator_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
        mode="startup",
    )

    # === Reset behaviors ===
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)},  # Smaller reset range
            "velocity_range": {
                "x": (-0.1, 0.1),  # Much smaller initial velocities
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (-0.1, 0.1),  # Much smaller initial angular velocities
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),  # Closer to default positions
            "velocity_range": (0.0, 0.0),
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-0.5, 0.5),  # Much smaller disturbances
            "torque_range": (-0.5, 0.5),  # Much smaller disturbances
        },
        mode="reset",
    )

    # === Interval behaviors ===
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # === Reward for task-space performance ===
    # command tracking performance
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=10.0,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=1.0,
    )

    # === Reward for basic behaviors ===
    # termination penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,
    )

    # base motion smoothness
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.1,
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    # ensure the robot is standing upright
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
    )

    # joint motion smoothness
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BIPED_LOCO_JOINTS)},
        weight=-2.0e-3,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BIPED_LOCO_JOINTS)},
        weight=-1.0e-6,
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )

    # === Reward for encouraging behaviors ===
    # encourage robot to take steps
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.4,
        },
        weight=5.0,
    )
    # penalize feet sliding on the ground to exploit physics sim inaccuracies
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
        weight=-5.0,
    )

    # penalize undesired contacts
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_hip_.*", ".*_knee_.*"]),
            "threshold": 1.0,
        },
        weight=-1.0,
    )

    # penalize deviation from default of the joints that are not essential for locomotion
    # Note: This robot doesn't have hip yaw joints, so this reward term is not applicable
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"])},
        weight=-0.5,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.78, "asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )

@configclass
class CurriculumsCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class BipedLocoEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BipedLocoSceneCfg = BipedLocoSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()
    curriculum = CurriculumsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation