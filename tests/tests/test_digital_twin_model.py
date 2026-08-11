"""Structural validation for the Digital Twin v0.1 MJCF model."""

from pathlib import Path

import mujoco

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "mujoco"
    / "models"
    / "digital_twin_v0_1.xml"
)

ARTICULATED_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_roll",
    "left_ankle_pitch",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_roll",
    "right_ankle_pitch",
)

POSITION_ACTUATORS = (
    "left_hip_roll_position",
    "left_hip_pitch_position",
    "left_knee_pitch_position",
    "left_ankle_roll_position",
    "left_ankle_pitch_position",
    "right_hip_roll_position",
    "right_hip_pitch_position",
    "right_knee_pitch_position",
    "right_ankle_roll_position",
    "right_ankle_pitch_position",
)

ACTUATED_JOINTS = (
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_roll",
    "left_ankle_pitch",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_roll",
    "right_ankle_pitch",
)

EXPECTED_PARENT = {
    "pelvis": "pelvis_root",
    "trunk": "pelvis",
    "left_hip_yaw_link": "pelvis",
    "left_hip_roll_link": "left_hip_yaw_link",
    "left_thigh": "left_hip_roll_link",
    "left_shank": "left_thigh",
    "left_ankle": "left_shank",
    "left_foot": "left_ankle",
    "right_hip_yaw_link": "pelvis",
    "right_hip_roll_link": "right_hip_yaw_link",
    "right_thigh": "right_hip_roll_link",
    "right_shank": "right_thigh",
    "right_ankle": "right_shank",
    "right_foot": "right_ankle",
}

EXPECTED_JOINT = {
    "root_freejoint": ("pelvis_root", mujoco.mjtJoint.mjJNT_FREE, None),
    "left_hip_yaw": ("left_hip_yaw_link", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 0.0, 1.0)),
    "left_hip_roll": ("left_hip_roll_link", mujoco.mjtJoint.mjJNT_HINGE, (1.0, 0.0, 0.0)),
    "left_hip_pitch": ("left_thigh", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 1.0, 0.0)),
    "left_knee_pitch": ("left_shank", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 1.0, 0.0)),
    "left_ankle_roll": ("left_ankle", mujoco.mjtJoint.mjJNT_HINGE, (1.0, 0.0, 0.0)),
    "left_ankle_pitch": ("left_foot", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 1.0, 0.0)),
    "right_hip_yaw": ("right_hip_yaw_link", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 0.0, 1.0)),
    "right_hip_roll": ("right_hip_roll_link", mujoco.mjtJoint.mjJNT_HINGE, (1.0, 0.0, 0.0)),
    "right_hip_pitch": ("right_thigh", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 1.0, 0.0)),
    "right_knee_pitch": ("right_shank", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 1.0, 0.0)),
    "right_ankle_roll": ("right_ankle", mujoco.mjtJoint.mjJNT_HINGE, (1.0, 0.0, 0.0)),
    "right_ankle_pitch": ("right_foot", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 1.0, 0.0)),
}


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    assert name is not None
    return name


def test_model_loads_with_expected_degrees_of_freedom() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    assert model.njnt == 13
    assert model.nq == 19
    assert model.nv == 18
    assert model.nu == 10

    joint_names = {
        _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
    }
    assert joint_names == {"root_freejoint", *ARTICULATED_JOINTS}

    for joint_name, (body_name, joint_type, axis) in EXPECTED_JOINT.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert model.jnt_type[joint_id] == joint_type
        if axis is not None:
            assert tuple(model.jnt_axis[joint_id]) == axis

        body_id = int(model.jnt_bodyid[joint_id])
        assert _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) == body_name

    actuator_names = tuple(
        _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    )
    assert actuator_names == POSITION_ACTUATORS
    assert model.actuator_ctrllimited.all()

    actuator_joint_names = tuple(
        _name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator_id, 0]),
        )
        for actuator_id in range(model.nu)
    )
    assert actuator_joint_names == ACTUATED_JOINTS


def test_body_tree_matches_the_biped_kinematic_structure() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    for body_name, expected_parent_name in EXPECTED_PARENT.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert body_id >= 0, f"Missing body: {body_name}"

        parent_id = int(model.body_parentid[body_id])
        parent_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        assert parent_name == expected_parent_name
