import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Paramètres globaux
# -----------------------------
GAMMA = 0.99
C = 1.4
SIMULATIONS = 1000
MAX_STEPS = 100
EPISODES = 20

# -----------------------------
# Classe Node avec mémoire
# -----------------------------
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0

    def ucb1(self, child):
        if child.visits == 0:
            return float("inf")
        return (child.total_reward / child.visits) + C * np.sqrt(np.log(self.visits + 1) / child.visits)

    def best_child(self):
        return max(self.children.values(), key=lambda c: self.ucb1(c))

    def expand(self, env):
        for action in range(env.action_space.n):
            env_copy = gym.make("FrozenLake-v1", is_slippery=False)
            env_copy.reset()
            env_copy.unwrapped.s = self.state
            next_state, reward, terminated, truncated, _ = env_copy.step(action)
            if (action not in self.children):
                self.children[action] = Node(next_state, parent=self)

    def simulate(self, env):
        env_copy = gym.make("FrozenLake-v1", is_slippery=False)
        env_copy.reset()
        env_copy.unwrapped.s = self.state
        total_reward, discount = 0, 1.0
        for _ in range(MAX_STEPS):
            action = env_copy.action_space.sample()
            obs, reward, terminated, truncated, _ = env_copy.step(action)
            total_reward += reward * discount
            if terminated or truncated:
                break
            discount *= GAMMA
        return total_reward

    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

# -----------------------------
# MCTS avec arbre persistent
# -----------------------------
tree_memory = {}  # Dict[state -> Node]

def mcts(env, state):
    if state not in tree_memory:
        tree_memory[state] = Node(state)

    root = tree_memory[state]
    # Exécution de SIMULATIONS épisodes MCTS avec barre de progression
    for _ in range(SIMULATIONS):

        node = root
        # Sélection
        while node.children:
            node = node.best_child()

        # Expansion
        node.expand(env)

        # Simulation
        reward = node.simulate(env)

        # Backpropagation
        node.backpropagate(reward)

    best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_action

# -----------------------------
# Entraînement sur FrozenLake
# -----------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)
all_rewards = []

for episode in trange(EPISODES, desc="Training episodes"):
    state, _ = env.reset()
    total_reward = 0
    for _ in trange(MAX_STEPS, desc="Steps per episode", leave=False):
        action = mcts(env, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    all_rewards.append(total_reward)

# -----------------------------
# Affichage des résultats
# -----------------------------
import pandas as pd
import seaborn as sns


df = pd.DataFrame({"Episode": np.arange(1, EPISODES+1), "Reward": all_rewards})
df["Reward_moving_avg"] = df["Reward"].rolling(window=10).mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x="Episode", y="Reward_moving_avg", label="Reward (moving avg)", data=df)
plt.title("Phase 5 - MCTS avec mémoire persistante (FrozenLake)")
plt.xlabel("Épisode")
plt.ylabel("Reward")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("phase5_results.png")
plt.figure(figsize=(10, 5))
sns.lineplot(x="Episode", y="Reward_moving_avg", label="Reward (moving avg)", data=df)
plt.title("Phase 4 - MCTS avec mémoire persistante (FrozenLake)")
plt.xlabel("Épisode")
plt.ylabel("Reward")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("phase5_results.png")
