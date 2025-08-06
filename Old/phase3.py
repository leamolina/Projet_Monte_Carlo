import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def run_optimized_mcts(env_id, n_episodes=300, max_steps=100, gamma=0.99, c=1.0, simulations_per_action=30):
    env = gym.make(env_id)
    Q = defaultdict(float)
    N = defaultdict(int)
    Ns = defaultdict(int)

    def uct_value(state, action):
        if N[(state, action)] == 0:
            return float("inf")
        return (Q[(state, action)] / N[(state, action)]) + c * np.sqrt(
            np.log(Ns[state] + 1) / N[(state, action)]
        )

    def select_action(state):
        actions = list(range(env.action_space.n))
        uct_values = [uct_value(state, a) for a in actions]
        return int(np.argmax(uct_values))

    def simulate_from(state, action):
        sim_env = gym.make(env_id)
        sim_env.reset(seed=None)
        try:
            sim_env.unwrapped.s = state
        except AttributeError:
            return 0.0

        next_state, reward, terminated, truncated, _ = sim_env.step(action)
        total_reward = reward
        discount = gamma

        while not (terminated or truncated):
            a = sim_env.action_space.sample()
            next_state, reward, terminated, truncated, _ = sim_env.step(a)
            total_reward += reward * discount
            discount *= gamma

        return total_reward

    def mcts_episode():
        state, _ = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            for action in range(env.action_space.n):
                for _ in range(simulations_per_action):
                    r = simulate_from(state, action)
                    Q[(state, action)] += r
                    N[(state, action)] += 1
                    Ns[state] += 1

            a = select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(a)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        return total_reward

    rewards = []
    avg_rewards = []
    window = 50
    for _ in range(n_episodes):
        r = mcts_episode()
        rewards.append(r)
        avg_rewards.append(np.mean(rewards[-window:]))
    return avg_rewards

# Liste des environnements
games = ["FrozenLake-v1", "Taxi-v3", "CliffWalking-v0"]
results = {}

for game in games:
    print(f"Lancement de {game}...")
    try:
        results[game] = run_optimized_mcts(game)
    except Exception as e:
        print(f"Erreur pour {game} : {e}")
        results[game] = [0] * 300

# Tracé des courbes
plt.figure(figsize=(12, 6))
for game, rewards in results.items():
    plt.plot(rewards, label=game)
plt.title("Optimized MCTS — Comparaison multi-jeux (Phase 3)")
plt.xlabel("Épisodes")
plt.ylabel("Reward moyenne (fenêtre)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
