"""Minimal Gymnasium environment for the bipedal Digital Twin v0.1."""

from __future__ import annotations

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


class BipedalTwinEnv(gym.Env[FloatArray, FloatArray]):
    """Expose the Digital Twin state and normalized position targets."""

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

    def __init__(self, max_episode_steps: int = 1_000) -> None:
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.actuated_qpos_addresses] = self.nominal_pose
        self.data.qpos[self.root_qpos_address + 2] = self.NOMINAL_ROOT_HEIGHT
        self._apply_action(np.zeros(self.model.nu, dtype=np.float32))
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0

        info = {"nominal_pose": self.nominal_pose.copy()}

        return self._get_observation(), info

    def step(
        self,
        action: FloatArray,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        observation = self._get_observation()
        terminated = not np.isfinite(observation).all()
        truncated = self.current_step >= self.max_episode_steps
        info = {"simulation_time": float(self.data.time)}

        return observation, 0.0, terminated, truncated, info

    def close(self) -> None:
        """Release references owned by the environment."""

        self.data = None  # type: ignore[assignment]
        self.model = None  # type: ignore[assignment]
