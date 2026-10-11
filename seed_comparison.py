from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


ENV_ID = "Acrobot-v1"
SEEDS = [42, 25, 123]
N_EVAL_EPISODES = 30

HYPERPARAMS = {
    "learning_rate": 6.3e-4,
    "buffer_size": 50_000,
    "learning_starts": 0,
    "batch_size": 128,
    "gamma": 0.99,
    "target_update_interval": 250,
    "train_freq": 4,
    "gradient_steps": -1,
    "exploration_fraction": 0.12,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.10,
    "net_arch": [256, 256],
    "total_timesteps": 100_000,
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
SEED_RESULTS_DIR = RESULTS_DIR / "seed_comparison"
SEED_RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, seed):
    returns = []
    lengths = []
    successes = []

    for episode in range(N_EVAL_EPISODES):
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(int(terminated))
        env.close()

    return pd.DataFrame(
        {
            "seed": seed,
            "episode": np.arange(1, N_EVAL_EPISODES + 1),
            "return": returns,
            "length": lengths,
            "success": successes,
        }
    )


def train_for_seed(seed):
    log_dir = SEED_RESULTS_DIR / f"logs_seed_{seed}"
    log_dir.mkdir(exist_ok=True)

    env = Monitor(gym.make(ENV_ID), str(log_dir))
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=HYPERPARAMS["learning_rate"],
        buffer_size=HYPERPARAMS["buffer_size"],
        learning_starts=HYPERPARAMS["learning_starts"],
        batch_size=HYPERPARAMS["batch_size"],
        gamma=HYPERPARAMS["gamma"],
        target_update_interval=HYPERPARAMS["target_update_interval"],
        train_freq=HYPERPARAMS["train_freq"],
        gradient_steps=HYPERPARAMS["gradient_steps"],
        exploration_fraction=HYPERPARAMS["exploration_fraction"],
        exploration_initial_eps=HYPERPARAMS["exploration_initial_eps"],
        exploration_final_eps=HYPERPARAMS["exploration_final_eps"],
        policy_kwargs={"net_arch": HYPERPARAMS["net_arch"]},
        seed=seed,
        verbose=0,
        device="auto",
    )
    model.learn(total_timesteps=HYPERPARAMS["total_timesteps"], progress_bar=False)
    env.close()

    return evaluate_model(model, seed)


def main():
    evaluation_frames = []
    for seed in SEEDS:
        print(f"Training seed {seed}...")
        evaluation_frames.append(train_for_seed(seed))

    evaluation_df = pd.concat(evaluation_frames, ignore_index=True)
    evaluation_df.to_csv(SEED_RESULTS_DIR / "seed_evaluation_episodes.csv", index=False)

    summary_df = (
        evaluation_df.groupby("seed")
        .agg(
            retorno_promedio=("return", "mean"),
            desviacion_estandar=("return", "std"),
            retorno_minimo=("return", "min"),
            retorno_maximo=("return", "max"),
            longitud_promedio=("length", "mean"),
            tasa_exito=("success", "mean"),
        )
        .reset_index()
    )
    summary_df.to_csv(SEED_RESULTS_DIR / "seed_summary.csv", index=False)

    aggregate_df = pd.DataFrame(
        [
            {
                "semillas": ",".join(str(seed) for seed in SEEDS),
                "retorno_promedio_entre_semillas": summary_df["retorno_promedio"].mean(),
                "retorno_std_entre_semillas": summary_df["retorno_promedio"].std(ddof=1),
                "longitud_promedio_entre_semillas": summary_df["longitud_promedio"].mean(),
                "tasa_exito_promedio": summary_df["tasa_exito"].mean(),
            }
        ]
    )
    aggregate_df.to_csv(SEED_RESULTS_DIR / "seed_aggregate.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["seed"].astype(str), summary_df["retorno_promedio"])
    plt.title("Comparación de DQN por semilla")
    plt.xlabel("Semilla")
    plt.ylabel("Retorno promedio")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(SEED_RESULTS_DIR / "seed_return_comparison.png", dpi=300, bbox_inches="tight")

    print("\nResumen por semilla")
    print(summary_df.to_string(index=False))
    print("\nResumen agregado")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
