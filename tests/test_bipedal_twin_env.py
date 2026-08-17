"""Contract tests for the minimal BipedalTwinEnv."""

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from companion_robot.envs import BipedalTwinEnv, ResetPerturbationConfig


def test_environment_contract() -> None:
    env = BipedalTwinEnv(max_episode_steps=10)

    assert env.action_space.shape == (10,)
    check_env(env, skip_render_check=True)

    env.close()


def test_zero_action_maps_exactly_to_nominal_pose() -> None:
    env = BipedalTwinEnv(
        max_episode_steps=2,
        reset_config=ResetPerturbationConfig(enabled=False),
    )
    _, reset_info = env.reset(seed=0)

    np.testing.assert_array_equal(
        env.data.qpos[env.actuated_qpos_addresses],
        env.nominal_pose,
    )
    np.testing.assert_array_equal(env.data.ctrl, env.nominal_pose)
    np.testing.assert_array_equal(reset_info["nominal_pose"], env.nominal_pose)

    observation, reward, terminated, truncated, info = env.step(
        np.zeros(10, dtype=np.float32)
    )

    np.testing.assert_array_equal(env.data.ctrl, env.nominal_pose)
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert reward == info["reward_total"]
    assert not terminated
    assert not truncated
    assert info["simulation_time"] > 0.0

    env.close()


def test_extreme_actions_respect_physical_joint_limits() -> None:
    env = BipedalTwinEnv()

    negative_targets = env._action_to_targets(
        -np.ones(env.model.nu, dtype=np.float32)
    )
    positive_targets = env._action_to_targets(
        np.ones(env.model.nu, dtype=np.float32)
    )
    joint_ranges = env.model.jnt_range[env.actuated_joint_ids]

    np.testing.assert_allclose(negative_targets, env.target_low)
    np.testing.assert_allclose(positive_targets, env.target_high)
    assert np.all(negative_targets >= joint_ranges[:, 0])
    assert np.all(negative_targets <= joint_ranges[:, 1])
    assert np.all(positive_targets >= joint_ranges[:, 0])
    assert np.all(positive_targets <= joint_ranges[:, 1])

    env.close()


def test_nominal_pose_and_action_scales_are_symmetric() -> None:
    env = BipedalTwinEnv()

    np.testing.assert_allclose(env.nominal_pose[:5], env.nominal_pose[5:])
    np.testing.assert_allclose(
        env.negative_action_scale[:5],
        env.negative_action_scale[5:],
    )
    np.testing.assert_allclose(
        env.positive_action_scale[:5],
        env.positive_action_scale[5:],
    )
    np.testing.assert_allclose(
        np.rad2deg(env.nominal_pose),
        (0.0, -5.0, 10.0, 0.0, -5.0) * 2,
    )

    env.close()


def test_nominal_pose_keeps_trunk_vertical_and_soles_on_ground() -> None:
    env = BipedalTwinEnv(reset_config=ResetPerturbationConfig(enabled=False))
    env.reset(seed=0)

    trunk_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "trunk",
    )
    trunk_rotation = env.data.xmat[trunk_id].reshape(3, 3)
    np.testing.assert_allclose(trunk_rotation, np.eye(3), atol=1e-7)

    for foot_geom_name in ("left_foot_geom", "right_foot_geom"):
        geom_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            foot_geom_name,
        )
        rotation = env.data.geom_xmat[geom_id].reshape(3, 3)
        half_extents = env.model.geom_size[geom_id]
        vertical_extent = float(np.abs(rotation[2]) @ half_extents)
        sole_height = env.data.geom_xpos[geom_id, 2] - vertical_extent

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(sole_height, 0.0, atol=1e-6)

    env.close()
