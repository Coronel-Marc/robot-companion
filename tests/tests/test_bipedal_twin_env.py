"""Contract tests for the minimal BipedalTwinEnv."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from companion_robot.envs import BipedalTwinEnv


def test_environment_contract() -> None:
    env = BipedalTwinEnv(max_episode_steps=10)

    check_env(env, skip_render_check=True)

    env.close()


def test_normalized_actions_map_to_actuator_control_ranges() -> None:
    env = BipedalTwinEnv(max_episode_steps=2)
    env.reset(seed=0)

    action = np.linspace(-1.0, 1.0, env.model.nu, dtype=np.float32)
    expected_control = env.model.actuator_ctrlrange[:, 0] + 0.5 * (action + 1.0) * (
        env.model.actuator_ctrlrange[:, 1] - env.model.actuator_ctrlrange[:, 0]
    )

    observation, reward, terminated, truncated, info = env.step(action)

    np.testing.assert_allclose(env.data.ctrl, expected_control)
    assert env.observation_space.contains(observation)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["simulation_time"] > 0.0

    env.close()
