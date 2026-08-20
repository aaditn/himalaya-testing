"""G1 23-DOF articulation config for Isaac Lab.

Joint order, limits, and effort figures below are read directly from
assets/g1/g1_23dof.urdf -- not guessed. Effort ceilings that matter:

    knee    139 Nm
    hip      88 Nm
    waist    88 Nm  (yaw only -- this variant has no waist pitch/roll)
    ankle    35 Nm
    arm      25 Nm   <- 5.6x weaker than the knee

That arm/knee ratio is the quantitative reason arms may not recruit for
balance on their own: the legs are far cheaper actuators for the same
corrective impulse. See ANGULAR_MOMENTUM in the reward config.

PD gains follow unitree_rl_gym's published G1 config for the legs. That
config locks the upper body, so it specifies NO arm gains -- the arm values
here are ours, scaled to the weaker arm actuators.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Actuated joints, in URDF order. Index ranges below depend on this order.
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
]
LEG_JOINTS = JOINT_NAMES[:12]
WAIST_JOINTS = JOINT_NAMES[12:13]
ARM_JOINTS = JOINT_NAMES[13:]

assert len(JOINT_NAMES) == 23
assert len(ARM_JOINTS) == 10

# Nominal stance. Legs from unitree_rl_gym (slight knee bend -- a straight-leg
# start makes the first contacts stiff and the policy learns to lock knees).
# Arms hang with a small shoulder-roll offset so left/right are mirrored and
# the elbows are off their limits.
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.1, "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1, "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0, "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.9,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.9,
    "right_wrist_roll_joint": 0.0,
}

# URDF effort limits, verbatim. Used to sanity-check gains and to normalize
# the energy reward term so a 139 Nm knee and a 25 Nm shoulder contribute
# comparably rather than the knee dominating the penalty.
EFFORT_LIMITS = {
    **{j: 88.0 for j in JOINT_NAMES if "hip" in j or "waist" in j},
    **{j: 139.0 for j in JOINT_NAMES if "knee" in j},
    **{j: 35.0 for j in JOINT_NAMES if "ankle" in j},
    **{j: 25.0 for j in JOINT_NAMES if j in ARM_JOINTS},
}

G1_23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/g1/g1_23dof.usd",   # produced by scripts/convert_urdf.py
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
            enabled_self_collisions=True,   # arms can hit the torso; we penalize it
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.79),
        joint_pos=DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Leg gains: unitree_rl_gym published values.
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*"],
            effort_limit=88.0, velocity_limit=32.0,
            stiffness=100.0, damping=2.0,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=139.0, velocity_limit=20.0,
            stiffness=150.0, damping=4.0,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*"],
            effort_limit=35.0, velocity_limit=30.0,
            stiffness=40.0, damping=2.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit=88.0, velocity_limit=32.0,
            stiffness=100.0, damping=2.0,
        ),
        # Arm gains are OURS -- the reference config locks the upper body and
        # publishes none. Scaled down with the 25 Nm ceiling: gains high enough
        # to hold a pose against gravity, low enough that the arms stay
        # compliant and can actually swing. Stiff arms are dead weight, and
        # dead weight is the failure mode we are trying to avoid.
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*"],
            effort_limit=25.0, velocity_limit=37.0,
            stiffness=40.0, damping=2.0,
        ),
        "elbows_wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_elbow_joint", ".*_wrist_roll_joint"],
            effort_limit=25.0, velocity_limit=37.0,
            stiffness=20.0, damping=1.0,
        ),
    },
)
