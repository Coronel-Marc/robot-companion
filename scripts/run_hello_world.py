"""Executa uma simulação mínima do MuJoCo."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "mujoco" / "models" / "hello_world.xml"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()