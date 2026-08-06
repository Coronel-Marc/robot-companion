"""Simulação de uma perna simples controlada por servos de posição."""

from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "assets" / "mujoco" / "models" / "single_leg.xml"


def run() -> None:
    """Carrega o modelo e movimenta quadril e joelho."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo MJCF não encontrado: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    hip_actuator_id = model.actuator("hip_motor").id
    knee_actuator_id = model.actuator("knee_motor").id

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_started_at = time.perf_counter()

            movement_phase = data.time

            hip_target = 25.0 * math.sin(movement_phase * 1.5)
            knee_target = 45.0 + 30.0 * math.sin(movement_phase * 1.5 + 0.8)

            data.ctrl[hip_actuator_id] = hip_target
            data.ctrl[knee_actuator_id] = knee_target

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - step_started_at
            sleep_time = model.opt.timestep - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    run()