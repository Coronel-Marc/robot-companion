"""Posture, stability, and fall-detection tests for Stage 4."""

import mujoco
import numpy as np
import pytest

from companion_robot.envs import BipedalTwinEnv

DIAGNOSTIC_FLOAT_KEYS = (
    "reward_total",
    "reward_upright",
    "reward_height",
    "reward_pose",
    "penalty_velocity",
    "penalty_effort",
    "pelvis_height",
    "torso_tilt",
)


def _set_root_pitch(env: BipedalTwinEnv, angle_degrees: float) -> None:
    half_angle = np.deg2rad(angle_degrees) / 2.0
    quaternion_address = env.root_qpos_address + 3
    env.data.qpos[quaternion_address : quaternion_address + 4] = (
        np.cos(half_angle),
        0.0,
        np.sin(half_angle),
        0.0,
    )
    mujoco.mj_forward(env.model, env.data)


def test_nominal_pose_scores_better_than_a_strongly_tilted_pose() -> None:
    env = BipedalTwinEnv()
    env.reset(seed=0)
    nominal_metrics = env.evaluate_posture()

    _set_root_pitch(env, 55.0)
    tilted_metrics = env.evaluate_posture()

    assert nominal_metrics["reward_total"] > tilted_metrics["reward_total"]
    assert nominal_metrics["reward_upright"] > tilted_metrics["reward_upright"]

    env.close()


def test_lowering_pelvis_reduces_height_reward() -> None:
    env = BipedalTwinEnv()
    env.reset(seed=0)
    nominal_height_reward = env.evaluate_posture()["reward_height"]

    env.data.qpos[env.root_qpos_address + 2] = env.nominal_pelvis_height * 0.55
    mujoco.mj_forward(env.model, env.data)
    lowered_height_reward = env.evaluate_posture()["reward_height"]

    assert nominal_height_reward > lowered_height_reward

    env.close()


def test_increasing_torso_tilt_reduces_upright_reward() -> None:
    env = BipedalTwinEnv()
    env.reset(seed=0)

    _set_root_pitch(env, 15.0)
    slight_tilt = env.evaluate_posture()
    _set_root_pitch(env, 45.0)
    strong_tilt = env.evaluate_posture()

    assert slight_tilt["torso_tilt"] < strong_tilt["torso_tilt"]
    assert slight_tilt["reward_upright"] > strong_tilt["reward_upright"]

    env.close()


def test_joint_and_body_velocities_increase_velocity_penalty() -> None:
    env = BipedalTwinEnv()
    env.reset(seed=0)
    stationary_penalty = env.evaluate_posture()["penalty_velocity"]

    env.data.qvel[env.actuated_dof_addresses] = 4.0
    env.data.qvel[3:6] = (1.0, 2.0, 3.0)
    mujoco.mj_forward(env.model, env.data)
    moving_penalty = env.evaluate_posture()["penalty_velocity"]

    assert moving_penalty > stationary_penalty

    env.close()


def test_actuator_force_increases_effort_penalty() -> None:
    env = BipedalTwinEnv()
    env.reset(seed=0)
    nominal_penalty = env.evaluate_posture()["penalty_effort"]

    env.data.ctrl[:] = env.target_high
    mujoco.mj_forward(env.model, env.data)
    loaded_penalty = env.evaluate_posture()["penalty_effort"]

    assert np.linalg.norm(env.data.actuator_force) > 0.0
    assert loaded_penalty > nominal_penalty

    env.close()


def test_reward_components_are_finite_and_match_weighted_total() -> None:
    env = BipedalTwinEnv()
    env.reset(seed=0)
    env.data.qvel[env.actuated_dof_addresses] = np.linspace(-2.0, 2.0, env.model.nu)
    env.data.ctrl[:] = env.target_high
    mujoco.mj_forward(env.model, env.data)
    metrics = env.evaluate_posture()

    for key in DIAGNOSTIC_FLOAT_KEYS:
        assert np.isfinite(metrics[key]), key

    config = env.reward_config
    expected_total = (
        config.upright_weight * metrics["reward_upright"]
        + config.height_weight * metrics["reward_height"]
        + config.pose_weight * metrics["reward_pose"]
        - config.velocity_weight * metrics["penalty_velocity"]
        - config.effort_weight * metrics["penalty_effort"]
    )
    assert metrics["reward_total"] == pytest.approx(expected_total)

    env.close()


def test_low_pelvis_terminates_episode_as_a_fall() -> None:
    env = BipedalTwinEnv(max_episode_steps=10)
    env.reset(seed=0)
    env.data.qpos[env.root_qpos_address + 2] = env.minimum_pelvis_height - 0.02
    mujoco.mj_forward(env.model, env.data)

    _, _, terminated, truncated, info = env.step(np.zeros(env.model.nu, dtype=np.float32))

    assert terminated
    assert not truncated
    assert info["is_fallen"]

    env.close()


def test_excessive_torso_tilt_terminates_episode_as_a_fall() -> None:
    env = BipedalTwinEnv(max_episode_steps=10)
    env.reset(seed=0)
    _set_root_pitch(env, 70.0)

    _, _, terminated, truncated, info = env.step(np.zeros(env.model.nu, dtype=np.float32))

    assert terminated
    assert not truncated
    assert info["is_fallen"]

    env.close()


def test_non_finite_state_is_detected_and_metrics_remain_finite() -> None:
    env = BipedalTwinEnv()
    env.reset(seed=0)
    env.data.qvel[env.actuated_dof_addresses[0]] = np.nan

    metrics = env.evaluate_posture()

    assert metrics["is_fallen"]
    for key in DIAGNOSTIC_FLOAT_KEYS:
        assert np.isfinite(metrics[key]), key

    env.close()


def test_time_limit_alone_truncates_without_termination() -> None:
    env = BipedalTwinEnv(max_episode_steps=1)
    env.reset(seed=0)

    _, _, terminated, truncated, info = env.step(np.zeros(env.model.nu, dtype=np.float32))

    assert not terminated
    assert truncated
    assert not info["is_fallen"]

    env.close()


def test_ten_simulated_seconds_remain_numerically_stable_across_resets() -> None:
    env = BipedalTwinEnv(max_episode_steps=1_000)
    observation, _ = env.reset(seed=0)
    simulated_duration = 0.0
    reset_count = 0
    action = np.zeros(env.model.nu, dtype=np.float32)

    while simulated_duration < 10.0:
        observation, reward, terminated, truncated, info = env.step(action)
        simulated_duration += float(env.model.opt.timestep)

        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        for key in DIAGNOSTIC_FLOAT_KEYS:
            assert np.isfinite(info[key]), key
        assert reward == info["reward_total"]

        if terminated or truncated:
            observation, reset_info = env.reset()
            reset_count += 1
            assert np.isfinite(observation).all()
            assert not reset_info["is_fallen"]

    assert reset_count > 0

    env.close()
