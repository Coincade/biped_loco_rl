import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

BIPED_LOCO_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_roll_joint",
    "left_ankle_roll_joint",
    "left_foot_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_roll_joint",
    "right_ankle_roll_joint",
    "right_foot_joint",
]

BIPED_LOCO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/yash/Downloads/biped_loco_urdf/biped_loco.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_knee_roll_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "left_foot_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_knee_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "right_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_knee_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_knee_roll_joint",
            ],
            effort_limit=3,
            velocity_limit=4.37,
            stiffness=20,
            damping=0.4,
            armature=0.007,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_ankle_roll_joint",
                "left_foot_joint",
                "right_ankle_roll_joint",
                "right_foot_joint",
            ],
            effort_limit=3,
            velocity_limit=4.71,
            stiffness=20,
            damping=0.25,
            armature=0.002,
        ),
    },
)