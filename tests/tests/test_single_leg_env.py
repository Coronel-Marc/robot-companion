"""Testes do ambiente SingleLegEnv."""

from gymnasium.utils.env_checker import check_env

from companion_robot.envs import SingleLegEnv


def test_environment_contract() -> None:
    env = SingleLegEnv(max_episode_steps=10)

    check_env(env, skip_render_check=True)

    env.close()