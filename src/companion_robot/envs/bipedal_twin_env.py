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

    def _apply_action(self, action: FloatArray) -> None:
        normalized_action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )
        control_low = self.model.actuator_ctrlrange[:, 0]
        control_high = self.model.actuator_ctrlrange[:, 1]

        self.data.ctrl[:] = control_low + 0.5 * (normalized_action + 1.0) * (
            control_high - control_low
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0

        return self._get_observation(), {}

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
