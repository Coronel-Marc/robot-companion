"""Visualize the bipedal twin receiving a neutral normalized action."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from companion_robot.envs import BipedalTwinEnv


def _print_servo_targets(env: BipedalTwinEnv) -> None:
    print("Neutral action sent to the 10 position servos:")

    for actuator_id, target_radians in enumerate(env.data.ctrl):
        actuator_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        print(
            f"  {actuator_id:02d} {actuator_name:<29} "
            f"target={np.rad2deg(target_radians):7.2f} deg"
        )


def main() -> None:
    env = BipedalTwinEnv(max_episode_steps=2**31 - 1)
    neutral_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    try:
        env.reset(seed=0)
        env.step(neutral_action)
        _print_servo_targets(env)

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.25)
            viewer.cam.distance = 0.9
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -15.0

            while viewer.is_running():
                step_started_at = time.perf_counter()

                _, _, terminated, truncated, _ = env.step(neutral_action)
                viewer.sync()

                if terminated or truncated:
                    print(
                        "Simulation stopped:",
                        {"terminated": terminated, "truncated": truncated},
                    )
                    break

                remaining = env.model.opt.timestep - (
                    time.perf_counter() - step_started_at
                )
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
