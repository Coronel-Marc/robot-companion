"""Perturbed-reset and initial-condition robustness tests for Stage 5."""

import mujoco
import numpy as np
import pytest

from companion_robot.envs import BipedalTwinEnv, ResetPerturbationConfig

RESET_DIAGNOSTIC_FLOAT_KEYS = (
    "initial_joint_position_perturbation_norm",
    "initial_joint_position_perturbation_max_abs",
    "initial_root_roll",
    "initial_root_pitch",
    "initial_joint_velocity_norm",
    "initial_joint_velocity_max_abs",
    "initial_root_angular_velocity_norm",
)


def test_same_seed_reproduces_the_same_initial_state() -> None:
    env = BipedalTwinEnv()

    observation_a, info_a = env.reset(seed=42)
    qpos_a = env.data.qpos.copy()
    qvel_a = env.data.qvel.copy()
    observation_b, info_b = env.reset(seed=42)

    np.testing.assert_array_equal(observation_a, observation_b)
    np.testing.assert_array_equal(qpos_a, env.data.qpos)
    np.testing.assert_array_equal(qvel_a, env.data.qvel)
    for key in RESET_DIAGNOSTIC_FLOAT_KEYS:
        assert info_a[key] == info_b[key]

    env.close()


def test_different_seeds_produce_different_initial_states() -> None:
    env = BipedalTwinEnv()

    observation_a, _ = env.reset(seed=42)
    observation_b, _ = env.reset(seed=123)

    assert not np.array_equal(observation_a, observation_b)

    env.close()


@pytest.mark.parametrize(
    "reset_config",
    (
        ResetPerturbationConfig(enabled=False),
        ResetPerturbationConfig(
            joint_position_amplitude_radians=0.0,
            root_roll_amplitude_radians=0.0,
            root_pitch_amplitude_radians=0.0,
            joint_velocity_amplitude=0.0,
            root_angular_velocity_amplitude=0.0,
        ),
    ),
)
def test_disabled_or_zero_perturbations_restore_exact_nominal_reset(
    reset_config: ResetPerturbationConfig,
) -> None:
    env = BipedalTwinEnv(reset_config=reset_config)
    observation, info = env.reset(seed=42)

    np.testing.assert_array_equal(
        env.data.qpos[env.actuated_qpos_addresses],
        env.nominal_pose,
    )
    np.testing.assert_array_equal(
        env.data.qpos[env.root_qpos_address + 3 : env.root_qpos_address + 7],
        (1.0, 0.0, 0.0, 0.0),
    )
    np.testing.assert_array_equal(env.data.qvel, np.zeros(env.model.nv))
    np.testing.assert_array_equal(env.data.ctrl, env.nominal_pose)
    assert np.isfinite(observation).all()
    assert not info["reset_perturbations_enabled"]
    for key in RESET_DIAGNOSTIC_FLOAT_KEYS:
        assert info[key] == 0.0

    env.close()


@pytest.mark.parametrize(
    ("enabled_category", "expected_changed_state"),
    (
        ("joint_positions_enabled", "joint_positions"),
        ("root_orientation_enabled", "root_orientation"),
        ("joint_velocities_enabled", "joint_velocities"),
        ("root_angular_velocity_enabled", "root_angular_velocity"),
    ),
)
def test_each_perturbation_category_can_be_enabled_independently(
    enabled_category: str,
    expected_changed_state: str,
) -> None:
    category_flags = {
        "joint_positions_enabled": False,
        "root_orientation_enabled": False,
        "joint_velocities_enabled": False,
        "root_angular_velocity_enabled": False,
    }
    category_flags[enabled_category] = True
    env = BipedalTwinEnv(reset_config=ResetPerturbationConfig(**category_flags))
    env.reset(seed=7)

    joint_position_changed = bool(
        np.any(env.data.qpos[env.actuated_qpos_addresses] != env.nominal_pose)
    )
    root_orientation_changed = bool(
        np.any(
            env.data.qpos[
                env.root_qpos_address + 3 : env.root_qpos_address + 7
            ]
            != (1.0, 0.0, 0.0, 0.0)
        )
    )
    joint_velocity_changed = bool(
        np.any(env.data.qvel[env.actuated_dof_addresses] != 0.0)
    )
    root_angular_velocity_changed = bool(
        np.any(env.data.qvel[env.root_dof_address + 3 : env.root_dof_address + 6] != 0.0)
    )
    changed_states = {
        "joint_positions": joint_position_changed,
        "root_orientation": root_orientation_changed,
        "joint_velocities": joint_velocity_changed,
        "root_angular_velocity": root_angular_velocity_changed,
    }

    assert changed_states.pop(expected_changed_state)
    assert not any(changed_states.values())

    env.close()


def test_configured_perturbations_stay_within_their_amplitudes() -> None:
    config = ResetPerturbationConfig(
        joint_position_amplitude_radians=np.deg2rad(1.25),
        root_roll_amplitude_radians=np.deg2rad(0.5),
        root_pitch_amplitude_radians=np.deg2rad(0.75),
        joint_velocity_amplitude=0.03,
        root_angular_velocity_amplitude=0.02,
    )
    env = BipedalTwinEnv(reset_config=config)
    _, info = env.reset(seed=91)

    joint_position_delta = (
        env.data.qpos[env.actuated_qpos_addresses] - env.nominal_pose
    )
    joint_velocity = env.data.qvel[env.actuated_dof_addresses]
    root_angular_velocity = env.data.qvel[
        env.root_dof_address + 3 : env.root_dof_address + 6
    ]

    assert np.max(np.abs(joint_position_delta)) <= config.joint_position_amplitude_radians
    assert abs(info["initial_root_roll"]) <= config.root_roll_amplitude_radians
    assert abs(info["initial_root_pitch"]) <= config.root_pitch_amplitude_radians
    assert np.max(np.abs(joint_velocity)) <= config.joint_velocity_amplitude
    assert np.max(np.abs(root_angular_velocity[:2])) <= (
        config.root_angular_velocity_amplitude
    )
    assert root_angular_velocity[2] == 0.0

    env.close()


def test_structurally_invalid_perturbation_amplitudes_are_rejected() -> None:
    with pytest.raises(ValueError, match="joint-limit margin"):
        BipedalTwinEnv(
            reset_config=ResetPerturbationConfig(
                joint_position_amplitude_radians=np.deg2rad(20.0)
            )
        )

    with pytest.raises(ValueError, match="already fallen"):
        BipedalTwinEnv(
            reset_config=ResetPerturbationConfig(
                root_roll_amplitude_radians=np.deg2rad(50.0),
                root_pitch_amplitude_radians=np.deg2rad(50.0),
            )
        )


def test_one_thousand_default_resets_are_valid_and_physically_bounded() -> None:
    env = BipedalTwinEnv()
    config = env.reset_config
    floor_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    deepest_initial_contact = 0.0

    for seed in range(1_000):
        observation, info = env.reset(seed=seed)
        joint_positions = env.data.qpos[env.actuated_qpos_addresses]
        quaternion = env.data.qpos[
            env.root_qpos_address + 3 : env.root_qpos_address + 7
        ]

        assert np.isfinite(observation).all()
        assert np.all(joint_positions >= env.target_low)
        assert np.all(joint_positions <= env.target_high)
        assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1.0e-12)
        assert not info["is_fallen"]
        assert info["reset_perturbations_enabled"]
        assert info["initial_joint_position_perturbation_max_abs"] <= (
            config.joint_position_amplitude_radians
        )
        assert info["initial_joint_velocity_max_abs"] <= (
            config.joint_velocity_amplitude
        )
        assert abs(info["initial_root_roll"]) <= config.root_roll_amplitude_radians
        assert abs(info["initial_root_pitch"]) <= config.root_pitch_amplitude_radians

        for contact_id in range(env.data.ncon):
            contact = env.data.contact[contact_id]
            if floor_geom_id not in contact.geom:
                continue
            deepest_initial_contact = min(
                deepest_initial_contact,
                float(contact.dist),
            )

    # HIPOTESE TEMPORARIA de validacao: penetracao inicial acima de 1 cm seria
    # desproporcional para o robo de aproximadamente 50 cm.
    assert deepest_initial_contact >= -0.01

    env.close()


def test_short_rollouts_after_varied_resets_remain_numerically_stable() -> None:
    env = BipedalTwinEnv(max_episode_steps=50)
    action = np.zeros(env.model.nu, dtype=np.float32)

    for seed in range(20):
        observation, reset_info = env.reset(seed=seed)
        assert np.isfinite(observation).all()
        assert not reset_info["is_fallen"]

        for _ in range(50):
            observation, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(observation).all()
            assert np.isfinite(reward)
            assert all(
                np.isfinite(value)
                for key, value in info.items()
                if key != "is_fallen"
            )
            if terminated or truncated:
                assert terminated == info["is_fallen"]
                break

    env.close()
