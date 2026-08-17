"""Visualize the bipedal twin receiving a neutral normalized action."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from companion_robot.envs import BipedalTwinEnv, ResetPerturbationConfig

METRICS_REFRESH_SECONDS = 0.1
EPISODE_END_PAUSE_SECONDS = 1.0


def _print_servo_targets(env: BipedalTwinEnv) -> None:
    print("Neutral action sent to the nominal pose of the 10 position servos:")

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


def _format_metrics(info: dict[str, object]) -> str:
    return (
        f"reward_total={float(info['reward_total']): 8.4f} | "
        f"reward_upright={float(info['reward_upright']):.4f} | "
        f"reward_height={float(info['reward_height']):.4f} | "
        f"reward_pose={float(info['reward_pose']):.4f} | "
        f"penalty_velocity={float(info['penalty_velocity']): 8.4f} | "
        f"penalty_effort={float(info['penalty_effort']): 8.4f} | "
        f"pelvis_height={float(info['pelvis_height']):.4f} m | "
        f"torso_tilt={float(info['torso_tilt']):.4f} rad | "
        f"is_fallen={bool(info['is_fallen'])!s:<5}"
    )


def _print_metrics(info: dict[str, object]) -> None:
    print(f"\r{_format_metrics(info)}", end="", flush=True)


def _print_reset_diagnostics(episode_number: int, info: dict[str, object]) -> None:
    joint_perturbation = float(info["initial_joint_position_perturbation_max_abs"])
    root_roll = float(info["initial_root_roll"])
    root_pitch = float(info["initial_root_pitch"])
    print(
        f"Episode {episode_number} reset | "
        f"joint_max={joint_perturbation:.4f} rad "
        f"({np.rad2deg(joint_perturbation):.2f} deg) | "
        f"roll={root_roll:+.4f} rad ({np.rad2deg(root_roll):+.2f} deg) | "
        f"pitch={root_pitch:+.4f} rad ({np.rad2deg(root_pitch):+.2f} deg) | "
        f"joint_velocity_norm={float(info['initial_joint_velocity_norm']):.4f} rad/s | "
        "root_angular_velocity_norm="
        f"{float(info['initial_root_angular_velocity_norm']):.4f} rad/s"
    )


def main() -> None:
    env = BipedalTwinEnv(
        max_episode_steps=2**31 - 1,
        reset_config=ResetPerturbationConfig(),
    )
    neutral_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    episode_number = 1

    try:
        _, reset_info = env.reset(seed=0)
        _print_servo_targets(env)
        _print_reset_diagnostics(episode_number, reset_info)
        _print_metrics(reset_info)
        last_metrics_update = time.perf_counter()

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.25)
            viewer.cam.distance = 0.9
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -15.0

            while viewer.is_running():
                step_started_at = time.perf_counter()

                _, _, terminated, truncated, info = env.step(neutral_action)
                viewer.sync()

                now = time.perf_counter()
                if (
                    now - last_metrics_update >= METRICS_REFRESH_SECONDS
                    or terminated
                    or truncated
                ):
                    _print_metrics(info)
                    last_metrics_update = now

                if terminated or truncated:
                    ending = "fall" if terminated else "time limit"
                    print(
                        f"\nEpisode {episode_number} ended by {ending}; "
                        f"observing final pose for {EPISODE_END_PAUSE_SECONDS:.1f} s..."
                    )
                    time.sleep(EPISODE_END_PAUSE_SECONDS)
                    episode_number += 1
                    _, reset_info = env.reset()
                    viewer.sync()
                    _print_reset_diagnostics(episode_number, reset_info)
                    _print_metrics(reset_info)
                    last_metrics_update = time.perf_counter()
                    continue

                remaining = env.model.opt.timestep - (
                    time.perf_counter() - step_started_at
                )
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print()
    finally:
        env.close()


if __name__ == "__main__":
    main()
