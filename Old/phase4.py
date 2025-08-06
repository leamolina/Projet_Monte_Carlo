import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque

# --- Paramètres globaux ---
n_episodes = 100
max_steps = 100
n_simulations = 300
gamma = 0.99
c = 1.4
exploration_eps = 0.05  # pour ε-greedy sur l'action finale
window = 10

# --- Classe Noeud MCTS ---
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = dict()
        self.visits = 0
        self.value = 0.0

    def ucb_score(self, action, child):
        if child.visits == 0:
            return float("inf")
        return (child.value / child.visits) + c * np.sqrt(np.log(self.visits + 1) / child.visits)

    def best_child(self):
        return max(self.children.items(), key=lambda item: self.ucb_score(item[0], item[1]))[1]

# --- Fonction MCTS principale ---
def mcts(env, root_state):
    root = Node(root_state)

    for _ in range(n_simulations):
        node = root
        state = root_state
        env_sim = gym.make(env.spec.id)
        env_sim.reset(seed=None)
        env_sim.unwrapped.s = state

        terminated = False
        truncated = False
        depth = 0

        # Sélection + Expansion
        while node.children and depth < max_steps:
            action, child = max(node.children.items(), key=lambda item: node.ucb_score(item[0], item[1]))
            next_state, reward, terminated, truncated, _ = env_sim.step(action)
            node = child
            state = next_state
            if terminated or truncated:
                break
            env_sim.unwrapped.s = state
            depth += 1

        # Expansion (si pas terminal)
        if not terminated and not truncated:
            for action in range(env.action_space.n):
                env_sim2 = gym.make(env.spec.id)
                env_sim2.reset(seed=None)
                env_sim2.unwrapped.s = state
                next_state, reward, term, trunc, _ = env_sim2.step(action)
                if action not in node.children:
                    node.children[action] = Node(next_state, parent=node)

        # Simulation
        total_reward = 0
        discount = 1.0
        steps = 0
        while not (terminated or truncated) and steps < max_steps:
            a = env_sim.action_space.sample()
            next_state, reward, terminated, truncated, _ = env_sim.step(a)
            total_reward += discount * reward
            discount *= gamma
            steps += 1

        # Backpropagation
        while node:
            node.visits += 1
            node.value += total_reward
            node = node.parent

    # Choix final d'action (ε-greedy)
    if random.random() < exploration_eps:
        return env.action_space.sample()
    if not root.children:
        return env.action_space.sample()
    best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_action

# --- Entraînement sur un environnement ---
def run_env(env_name, env):
    avg_rewards = []
    all_rewards = deque(maxlen=window)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = mcts(env, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        all_rewards.append(total_reward)
        avg_rewards.append(np.mean(all_rewards))
        print(f"[{env_name}] Épisode {episode+1} — Reward: {total_reward:.2f}")

    return avg_rewards

# --- Jeux à comparer ---
env_names = ["FrozenLake-v1", "CartPole-v1", "LunarLander-v2"]
results = {}

for name in env_names:
    print(f"\n==> Exécution sur {name}")
    if name == "FrozenLake-v1":
        env = gym.make(name, render_mode=None, is_slippery=False)
    else:
        env = gym.make(name, render_mode=None)
    results[name] = run_env(name, env)
    env.close()

# --- Affichage final ---
for name, rewards in results.items():
    plt.plot(rewards, label=name)

plt.title("Phase 4 — MCTS avec arbre (UCT) sur plusieurs environnements")
plt.xlabel("Épisodes")
plt.ylabel("Reward moyenne (fenêtrée)")
plt.legend()
plt.grid()
plt.show()
