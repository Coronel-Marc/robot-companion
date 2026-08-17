"""Gymnasium environment for posture evaluation of the Digital Twin v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "assets" / "mujoco" / "models" / "digital_twin_v0_1.xml"

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class PostureRewardConfig:
    """Temporary hypotheses used to scale the posture reward components."""

    # HIPOTESES TEMPORARIAS: positive posture terms have comparable magnitudes,
    # while motion and effort act as preferences rather than dominating posture.
    upright_weight: float = 1.0
    height_weight: float = 0.5
    pose_weight: float = 0.25
    velocity_weight: float = 0.1
    effort_weight: float = 0.01

    # HIPOTESES TEMPORARIAS: 30 degrees distinguishes a noticeable trunk tilt;
    # 25% of nominal height strongly rejects a pelvis near the floor; 20 degrees
    # allows small joint deviations without prescribing an exact static pose.
    upright_scale_radians: float = np.deg2rad(30.0)
    height_scale_fraction: float = 0.25
    pose_scale_radians: float = np.deg2rad(20.0)

    # HIPOTESES TEMPORARIAS: characteristic, not hard-limit, scales for fast
    # motion (rad/s) and actuator force (N or N.m according to the transmission).
    joint_velocity_scale: float = 5.0
    angular_velocity_scale: float = 5.0
    actuator_force_scale: float = 20.0

    def __post_init__(self) -> None:
        weights = (
            self.upright_weight,
            self.height_weight,
            self.pose_weight,
            self.velocity_weight,
            self.effort_weight,
        )
        scales = (
            self.upright_scale_radians,
            self.height_scale_fraction,
            self.pose_scale_radians,
            self.joint_velocity_scale,
            self.angular_velocity_scale,
            self.actuator_force_scale,
        )
        if any(weight < 0.0 or not np.isfinite(weight) for weight in weights):
            raise ValueError("Reward weights must be finite and non-negative")
        if any(scale <= 0.0 or not np.isfinite(scale) for scale in scales):
            raise ValueError("Reward scales must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class FallDetectionConfig:
    """Temporary hypotheses that define a clearly invalid posture."""

    # HIPOTESES TEMPORARIAS: below half nominal pelvis height, or beyond a
    # 60-degree trunk tilt, the robot is clearly outside an upright posture.
    minimum_pelvis_height_fraction: float = 0.5
    maximum_torso_tilt_radians: float = np.deg2rad(60.0)

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_pelvis_height_fraction < 1.0:
            raise ValueError("minimum_pelvis_height_fraction must be between zero and one")
        if not 0.0 < self.maximum_torso_tilt_radians <= np.pi:
            raise ValueError("maximum_torso_tilt_radians must be in (0, pi]")


@dataclass(frozen=True, slots=True)
class ResetPerturbationConfig:
    """Temporary hypotheses for small, reproducible initial-state variations."""

    enabled: bool = True
    joint_positions_enabled: bool = True
    root_orientation_enabled: bool = True
    joint_velocities_enabled: bool = True
    root_angular_velocity_enabled: bool = True

    # HIPOTESES TEMPORARIAS: two degrees around each actuated joint are small
    # relative to its range and keep every nominal joint away from its limits.
    joint_position_amplitude_radians: float = np.deg2rad(2.0)

    # HIPOTESES TEMPORARIAS: one degree of roll/pitch keeps the trunk close to
    # vertical while representing ordinary placement imperfections. Yaw is zero.
    root_roll_amplitude_radians: float = np.deg2rad(1.0)
    root_pitch_amplitude_radians: float = np.deg2rad(1.0)

    # HIPOTESES TEMPORARIAS: 0.1 rad/s at joints and 0.05 rad/s at root roll/pitch
    # are minor initial motion, not an external push. Linear and yaw speeds stay zero.
    joint_velocity_amplitude: float = 0.1
    root_angular_velocity_amplitude: float = 0.05

    def __post_init__(self) -> None:
        amplitudes = (
            self.joint_position_amplitude_radians,
            self.root_roll_amplitude_radians,
            self.root_pitch_amplitude_radians,
            self.joint_velocity_amplitude,
            self.root_angular_velocity_amplitude,
        )
        if any(amplitude < 0.0 or not np.isfinite(amplitude) for amplitude in amplitudes):
            raise ValueError("Reset perturbation amplitudes must be finite and non-negative")


class BipedalTwinEnv(gym.Env[FloatArray, FloatArray]):
    """Expose position control and inspectable posture-quality metrics."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    ACTUATED_JOINT_NAMES: ClassVar[tuple[str, ...]] = (
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

    # HIPÓTESE TEMPORÁRIA: flexão sagital suave e simétrica. Hip, knee e
    # ankle pitch somam zero em cada perna para manter os pés horizontais.
    NOMINAL_POSE_DEGREES: ClassVar[tuple[float, ...]] = (
        0.0,
        -5.0,
        10.0,
        0.0,
        -5.0,
        0.0,
        -5.0,
        10.0,
        0.0,
        -5.0,
    )

    # HIPÓTESE TEMPORÁRIA: altura calculada com as dimensões atuais para
    # deixar a face inferior dos dois pés aproximadamente em z=0.
    NOMINAL_ROOT_HEIGHT: ClassVar[float] = 0.292_125

    def __init__(
        self,
        max_episode_steps: int = 1_000,
        *,
        reward_config: PostureRewardConfig | None = None,
        fall_config: FallDetectionConfig | None = None,
        reset_config: ResetPerturbationConfig | None = None,
    ) -> None:
        super().__init__()

        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be greater than zero")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"MJCF model not found: {MODEL_PATH}")

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        if not np.all(self.model.actuator_ctrllimited):
            raise ValueError("All actuators must define ctrlrange")

        self.actuated_joint_ids = self.model.actuator_trnid[:, 0].astype(int)
        actuated_joint_names = tuple(
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(joint_id),
            )
            for joint_id in self.actuated_joint_ids
        )
        if actuated_joint_names != self.ACTUATED_JOINT_NAMES:
            raise ValueError(
                "Unexpected actuated joint order: "
                f"expected {self.ACTUATED_JOINT_NAMES}, got {actuated_joint_names}"
            )

        if not np.all(self.model.jnt_limited[self.actuated_joint_ids]):
            raise ValueError("All actuated joints must define physical limits")

        self.actuated_qpos_addresses = self.model.jnt_qposadr[
            self.actuated_joint_ids
        ].astype(int)
        self.actuated_dof_addresses = self.model.jnt_dofadr[
            self.actuated_joint_ids
        ].astype(int)
        joint_ranges = self.model.jnt_range[self.actuated_joint_ids]
        actuator_ranges = self.model.actuator_ctrlrange
        self.target_low = np.maximum(joint_ranges[:, 0], actuator_ranges[:, 0])
        self.target_high = np.minimum(joint_ranges[:, 1], actuator_ranges[:, 1])

        self.nominal_pose = np.deg2rad(
            np.asarray(self.NOMINAL_POSE_DEGREES, dtype=np.float64)
        )
        if self.nominal_pose.shape != (self.model.nu,):
            raise ValueError("Nominal pose must define one value per actuator")
        if np.any(self.nominal_pose < self.target_low) or np.any(
            self.nominal_pose > self.target_high
        ):
            raise ValueError("Nominal pose exceeds an actuated joint limit")

        self.negative_action_scale = self.nominal_pose - self.target_low
        self.positive_action_scale = self.target_high - self.nominal_pose

        root_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "root_freejoint",
        )
        if root_joint_id < 0:
            raise ValueError("Model does not contain root_freejoint")
        self.root_qpos_address = int(self.model.jnt_qposadr[root_joint_id])
        self.root_dof_address = int(self.model.jnt_dofadr[root_joint_id])

        self.pelvis_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )
        self.trunk_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "trunk",
        )
        if self.pelvis_body_id < 0 or self.trunk_body_id < 0:
            raise ValueError("Model must contain pelvis and trunk bodies")

        self.reward_config = reward_config or PostureRewardConfig()
        self.fall_config = fall_config or FallDetectionConfig()
        self.reset_config = reset_config or ResetPerturbationConfig()
        self._validate_reset_config()
        self.nominal_pelvis_height = self._calculate_nominal_pelvis_height()
        self.minimum_pelvis_height = (
            self.nominal_pelvis_height
            * self.fall_config.minimum_pelvis_height_fraction
        )

        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.model.nq + self.model.nv,),
            dtype=np.float32,
        )

    def _get_observation(self) -> FloatArray:
        return np.concatenate((self.data.qpos, self.data.qvel)).astype(np.float32)

    def _action_to_targets(self, action: FloatArray) -> NDArray[np.float64]:
        normalized_action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )
        action_scale = np.where(
            normalized_action < 0.0,
            self.negative_action_scale,
            self.positive_action_scale,
        )
        targets = self.nominal_pose + normalized_action * action_scale

        return np.clip(targets, self.target_low, self.target_high)

    def _apply_action(self, action: FloatArray) -> None:
        self.data.ctrl[:] = self._action_to_targets(action)

    def _set_nominal_state(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetData(self.model, data)
        data.qpos[self.actuated_qpos_addresses] = self.nominal_pose
        data.qpos[self.root_qpos_address + 2] = self.NOMINAL_ROOT_HEIGHT
        data.ctrl[:] = self.nominal_pose
        mujoco.mj_forward(self.model, data)

    def _calculate_nominal_pelvis_height(self) -> float:
        nominal_data = mujoco.MjData(self.model)
        self._set_nominal_state(nominal_data)
        height = float(nominal_data.xpos[self.pelvis_body_id, 2])
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError("Nominal pose must produce a finite, positive pelvis height")
        return height

    def _validate_reset_config(self) -> None:
        config = self.reset_config
        if not config.enabled:
            return

        if config.joint_positions_enabled:
            smallest_nominal_joint_margin = float(
                np.min(
                    np.minimum(
                        self.nominal_pose - self.target_low,
                        self.target_high - self.nominal_pose,
                    )
                )
            )
            if config.joint_position_amplitude_radians > smallest_nominal_joint_margin:
                raise ValueError(
                    "Joint position perturbation exceeds a nominal joint-limit margin"
                )

        if not config.root_orientation_enabled:
            return

        if (
            config.root_roll_amplitude_radians >= np.pi / 2.0
            or config.root_pitch_amplitude_radians >= np.pi / 2.0
        ):
            raise ValueError("Enabled root orientation amplitudes must be less than pi/2")

        minimum_vertical_alignment = (
            np.cos(config.root_roll_amplitude_radians)
            * np.cos(config.root_pitch_amplitude_radians)
        )
        maximum_initial_tilt = float(
            np.arccos(np.clip(minimum_vertical_alignment, -1.0, 1.0))
        )
        if maximum_initial_tilt >= self.fall_config.maximum_torso_tilt_radians:
            raise ValueError(
                "Root orientation perturbation can initialize an already fallen robot"
            )

    @staticmethod
    def _quaternion_from_roll_pitch(roll: float, pitch: float) -> NDArray[np.float64]:
        half_roll = roll / 2.0
        half_pitch = pitch / 2.0
        cos_roll = np.cos(half_roll)
        sin_roll = np.sin(half_roll)
        cos_pitch = np.cos(half_pitch)
        sin_pitch = np.sin(half_pitch)
        quaternion = np.asarray(
            (
                cos_pitch * cos_roll,
                cos_pitch * sin_roll,
                sin_pitch * cos_roll,
                -sin_pitch * sin_roll,
            ),
            dtype=np.float64,
        )
        return quaternion / np.linalg.norm(quaternion)

    def _apply_reset_perturbations(self) -> dict[str, float | bool]:
        config = self.reset_config
        joint_position_perturbation = np.zeros(self.model.nu, dtype=np.float64)
        joint_velocity = np.zeros(self.model.nu, dtype=np.float64)
        root_roll = 0.0
        root_pitch = 0.0
        root_angular_velocity = np.zeros(3, dtype=np.float64)

        joint_positions_perturbed = bool(
            config.enabled
            and config.joint_positions_enabled
            and config.joint_position_amplitude_radians > 0.0
        )
        if joint_positions_perturbed:
            joint_position_perturbation = self.np_random.uniform(
                -config.joint_position_amplitude_radians,
                config.joint_position_amplitude_radians,
                size=self.model.nu,
            )
            perturbed_positions = self.nominal_pose + joint_position_perturbation
            self.data.qpos[self.actuated_qpos_addresses] = perturbed_positions

        root_orientation_perturbed = bool(
            config.enabled
            and config.root_orientation_enabled
            and (
                config.root_roll_amplitude_radians > 0.0
                or config.root_pitch_amplitude_radians > 0.0
            )
        )
        if root_orientation_perturbed:
            root_roll = float(
                self.np_random.uniform(
                    -config.root_roll_amplitude_radians,
                    config.root_roll_amplitude_radians,
                )
            )
            root_pitch = float(
                self.np_random.uniform(
                    -config.root_pitch_amplitude_radians,
                    config.root_pitch_amplitude_radians,
                )
            )
            quaternion_address = self.root_qpos_address + 3
            self.data.qpos[quaternion_address : quaternion_address + 4] = (
                self._quaternion_from_roll_pitch(root_roll, root_pitch)
            )

        joint_velocities_perturbed = bool(
            config.enabled
            and config.joint_velocities_enabled
            and config.joint_velocity_amplitude > 0.0
        )
        if joint_velocities_perturbed:
            joint_velocity = self.np_random.uniform(
                -config.joint_velocity_amplitude,
                config.joint_velocity_amplitude,
                size=self.model.nu,
            )
            self.data.qvel[self.actuated_dof_addresses] = joint_velocity

        root_angular_velocity_perturbed = bool(
            config.enabled
            and config.root_angular_velocity_enabled
            and config.root_angular_velocity_amplitude > 0.0
        )
        if root_angular_velocity_perturbed:
            root_angular_velocity[:2] = self.np_random.uniform(
                -config.root_angular_velocity_amplitude,
                config.root_angular_velocity_amplitude,
                size=2,
            )
            angular_velocity_address = self.root_dof_address + 3
            self.data.qvel[
                angular_velocity_address : angular_velocity_address + 3
            ] = root_angular_velocity

        perturbations_enabled = bool(
            joint_positions_perturbed
            or root_orientation_perturbed
            or joint_velocities_perturbed
            or root_angular_velocity_perturbed
        )
        return {
            "reset_perturbations_enabled": perturbations_enabled,
            "reset_joint_positions_perturbed": joint_positions_perturbed,
            "reset_root_orientation_perturbed": root_orientation_perturbed,
            "reset_joint_velocities_perturbed": joint_velocities_perturbed,
            "reset_root_angular_velocity_perturbed": root_angular_velocity_perturbed,
            "initial_joint_position_perturbation_norm": float(
                np.linalg.norm(joint_position_perturbation)
            ),
            "initial_joint_position_perturbation_max_abs": float(
                np.max(np.abs(joint_position_perturbation))
            ),
            "initial_root_roll": root_roll,
            "initial_root_pitch": root_pitch,
            "initial_joint_velocity_norm": float(np.linalg.norm(joint_velocity)),
            "initial_joint_velocity_max_abs": float(np.max(np.abs(joint_velocity))),
            "initial_root_angular_velocity_norm": float(
                np.linalg.norm(root_angular_velocity)
            ),
        }

    def _state_is_finite(self) -> bool:
        return bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.xpos[self.pelvis_body_id]).all()
            and np.isfinite(self.data.xmat[self.trunk_body_id]).all()
        )

    def evaluate_posture(self) -> dict[str, float | bool]:
        """Return the independently inspectable reward and fall metrics."""

        state_is_finite = self._state_is_finite()
        pelvis_height_raw = float(self.data.xpos[self.pelvis_body_id, 2])
        pelvis_height = pelvis_height_raw if np.isfinite(pelvis_height_raw) else 0.0

        trunk_rotation = self.data.xmat[self.trunk_body_id].reshape(3, 3)
        vertical_alignment = float(trunk_rotation[2, 2])
        if np.isfinite(vertical_alignment):
            torso_tilt = float(np.arccos(np.clip(vertical_alignment, -1.0, 1.0)))
        else:
            torso_tilt = float(np.pi)

        config = self.reward_config
        upright_reward = float(np.exp(-((torso_tilt / config.upright_scale_radians) ** 2)))

        height_scale = self.nominal_pelvis_height * config.height_scale_fraction
        height_error = (pelvis_height - self.nominal_pelvis_height) / height_scale
        height_reward = float(np.exp(-(height_error**2)))

        joint_positions = self.data.qpos[self.actuated_qpos_addresses]
        if np.isfinite(joint_positions).all():
            pose_error_rms = float(np.sqrt(np.mean((joint_positions - self.nominal_pose) ** 2)))
            pose_reward = float(np.exp(-((pose_error_rms / config.pose_scale_radians) ** 2)))
        else:
            pose_reward = 0.0

        joint_velocities = self.data.qvel[self.actuated_dof_addresses]
        trunk_angular_velocity = self.data.cvel[self.trunk_body_id, :3]
        if np.isfinite(joint_velocities).all() and np.isfinite(trunk_angular_velocity).all():
            velocity_penalty = float(
                np.mean((joint_velocities / config.joint_velocity_scale) ** 2)
                + np.mean((trunk_angular_velocity / config.angular_velocity_scale) ** 2)
            )
        else:
            velocity_penalty = 1.0e6

        actuator_force = self.data.actuator_force
        if np.isfinite(actuator_force).all():
            effort_penalty = float(
                np.mean((actuator_force / config.actuator_force_scale) ** 2)
            )
        else:
            effort_penalty = 1.0e6

        is_fallen = bool(
            not state_is_finite
            or pelvis_height < self.minimum_pelvis_height
            or torso_tilt > self.fall_config.maximum_torso_tilt_radians
        )

        reward_total = float(
            config.upright_weight * upright_reward
            + config.height_weight * height_reward
            + config.pose_weight * pose_reward
            - config.velocity_weight * velocity_penalty
            - config.effort_weight * effort_penalty
        )

        return {
            "reward_total": reward_total,
            "reward_upright": upright_reward,
            "reward_height": height_reward,
            "reward_pose": pose_reward,
            "penalty_velocity": velocity_penalty,
            "penalty_effort": effort_penalty,
            "pelvis_height": pelvis_height,
            "torso_tilt": torso_tilt,
            "is_fallen": is_fallen,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self._set_nominal_state(self.data)
        reset_diagnostics = self._apply_reset_perturbations()
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0

        observation = self._get_observation()
        if not np.isfinite(observation).all():
            raise RuntimeError("Reset perturbations produced a non-finite state")

        posture_metrics = self.evaluate_posture()
        if posture_metrics["is_fallen"]:
            raise RuntimeError("Reset perturbations initialized an already fallen robot")

        info = {
            "nominal_pose": self.nominal_pose.copy(),
            **reset_diagnostics,
            **posture_metrics,
        }

        return observation, info

    def step(
        self,
        action: FloatArray,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        observation = self._get_observation()
        metrics = self.evaluate_posture()
        reward = float(metrics["reward_total"])
        terminated = bool(metrics["is_fallen"])
        truncated = self.current_step >= self.max_episode_steps and not terminated
        info = {
            "simulation_time": float(self.data.time),
            **metrics,
        }

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Release references owned by the environment."""

        self.data = None  # type: ignore[assignment]
        self.model = None  # type: ignore[assignment]
