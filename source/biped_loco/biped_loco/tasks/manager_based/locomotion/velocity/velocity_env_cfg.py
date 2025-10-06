# Copyright (c) 2024, Biped Loco Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs.mdp import *
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    # ground
    ground = AssetBaseCfg(
        prim_path="/World/defaults/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(8.0, 8.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )

    # robot
    robot: ArticulationCfg = MISSING

    # height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=False,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=[2.0, 2.0],
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/defaults/GroundPlane"],
    )

    # contact forces
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.002,
        history_length=2,
        debug_vis=False,
        filter_prim_paths_expr=["/World/defaults/GroundPlane"],
    )

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/defaults/Terrain",
        terrain_type="generator",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
        terrain_generator=ROUGH_TERRAINS_CFG,
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=last_action)
        height_scan = ObsTerm(
            func=height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    reset_robot_joints = EventTerm(
        func=reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # interval
    # Note: Some interval events are disabled as they don't exist in IsaacLab
    pass


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = NormalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=NormalVelocityCommandCfg.Ranges(
            mean_vel=(0.5, 0.0, 0.0),  # mean linear-x, linear-y, angular-z
            std_vel=(0.3, 0.3, 0.5),  # std linear-x, linear-y, angular-z
        ),
    )


@configclass
class RewardsCfg:
    """Reward specifications for the MDP."""

    # tracking
    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity"},
    )
    track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity"},
    )

    # penalties
    lin_vel_z_l2 = RewTerm(func=lin_vel_z_l2, weight=-0.2)
    undesired_contacts = RewTerm(
        func=undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base.*")},
    )
    flat_orientation_l2 = RewTerm(func=flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(
        func=joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*"])},
    )
    dof_torques_l2 = RewTerm(
        func=joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])},
    )

    # rewards
    feet_air_time = RewTerm(
        func=feet_air_time,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*")},
    )

    # penalties
    base_contact = RewTerm(
        func=illegal_contact,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base.*"), "threshold": 1.0},
    )


@configclass
class TerminationsCfg:
    """Termination specifications for the MDP."""

    # time out
    time_out = DoneTerm(func=time_out, time_out=True)

    # base contact
    base_contact = DoneTerm(
        func=illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base.*"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum specifications for the MDP."""

    terrain_levels = CurrTerm(func=terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
