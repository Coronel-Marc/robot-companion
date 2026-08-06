"""Ambiente Gymnasium para controlar uma perna articulada no MuJoCo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "assets" / "mujoco" / "models" / "single_leg.xml"

FloatArray = NDArray[np.float32]


class SingleLegEnv(gym.Env[FloatArray, FloatArray]):
    """Ambiente mínimo para controlar quadril e joelho."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1_000) -> None:
        super().__init__()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo MJCF não encontrado: {MODEL_PATH}")

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Ações normalizadas:
        # action[0] controla o quadril
        # action[1] controla o joelho
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # Observação:
        # posições das juntas + velocidades das juntas
        observation_size = self.model.nq + self.model.nv

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.hip_actuator_id = self.model.actuator("hip_motor").id
        self.knee_actuator_id = self.model.actuator("knee_motor").id

        self.target_pose = np.array(
            [
                0.0,
                np.deg2rad(45.0),
            ],
            dtype=np.float64,
        )

    def _get_observation(self) -> FloatArray:
        observation = np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
            ]
        )

        return observation.astype(np.float32)

    def _apply_action(self, action: FloatArray) -> None:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Limites físicos definidos no XML, convertidos para radianos.
        hip_min = np.deg2rad(-60.0)
        hip_max = np.deg2rad(60.0)

        knee_min = np.deg2rad(0.0)
        knee_max = np.deg2rad(110.0)

        hip_target = np.interp(action[0], [-1.0, 1.0], [hip_min, hip_max])
        knee_target = np.interp(action[1], [-1.0, 1.0], [knee_min, knee_max])

        self.data.ctrl[self.hip_actuator_id] = hip_target
        self.data.ctrl[self.knee_actuator_id] = knee_target

    def _calculate_reward(self) -> float:
        pose_error = self.data.qpos - self.target_pose

        position_penalty = float(np.sum(np.square(pose_error)))
        velocity_penalty = float(np.sum(np.square(self.data.qvel)))

        reward = 1.0
        reward -= 2.0 * position_penalty
        reward -= 0.01 * velocity_penalty

        return reward

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)

        del options

        mujoco.mj_resetData(self.model, self.data)

        initial_noise = self.np_random.uniform(
            low=-0.02,
            high=0.02,
            size=self.model.nq,
        )

        self.data.qpos[:] = self.target_pose + initial_noise
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0

        observation = self._get_observation()
        info = {"target_pose": self.target_pose.copy()}

        return observation, info

    def step(
        self,
        action: FloatArray,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        self._apply_action(action)

        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        observation = self._get_observation()
        reward = self._calculate_reward()

        terminated = not np.isfinite(observation).all()
        truncated = self.current_step >= self.max_episode_steps

        info = {
            "simulation_time": float(self.data.time),
            "pose_error": float(
                np.linalg.norm(self.data.qpos - self.target_pose)
            ),
        }

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Libera referências mantidas pelo ambiente."""

        self.data = None  # type: ignore[assignment]
        self.model = None  # type: ignore[assignment]