"""Executa ações aleatórias no ambiente da perna."""

from companion_robot.envs import SingleLegEnv


def main() -> None:
    env = SingleLegEnv(max_episode_steps=500)

    observation, info = env.reset(seed=42)

    print("Ambiente iniciado")
    print(f"Observação inicial: {observation}")
    print(f"Informações iniciais: {info}")
    print(f"Espaço de ações: {env.action_space}")
    print(f"Espaço de observações: {env.observation_space}")

    total_reward = 0.0

    for step_number in range(500):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if step_number % 100 == 0:
            print(
                f"Passo: {step_number:03d} | "
                f"Recompensa: {reward:8.3f} | "
                f"Erro: {info['pose_error']:.4f}"
            )

        if terminated or truncated:
            print(
                "Episódio encerrado:",
                {
                    "terminated": terminated,
                    "truncated": truncated,
                },
            )
            break

    print(f"Recompensa total: {total_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()