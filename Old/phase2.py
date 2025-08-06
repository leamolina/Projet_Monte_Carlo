import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

def q_learning_frozenlake(
    env_name="FrozenLake-v1",
    episodes=10000,
    alpha=0.8,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.9995
):
    env = gym.make(env_name, is_slippery=False)  # Environnement déterministe
    state_size = env.observation_space.n
    action_size = env.action_space.n

    Q = np.zeros((state_size, action_size))
    rewards = []
    epsilon = epsilon_start

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

    env.close()
    return Q, rewards

# --- Exécution et tracé des résultats ---

Q_table, episode_rewards = q_learning_frozenlake()

# Reward moyenne sur une fenêtre glissante
window = 100
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(moving_avg)
plt.title("Q-learning sur FrozenLake (reward moyenne)")
plt.xlabel("Épisodes")
plt.ylabel(f"Reward moyenne (fenêtre={window})")
plt.grid()
plt.show()

# --- Politique apprise (meilleure action par état) ---
actions = ['←', '↓', '→', '↑']
policy = np.argmax(Q_table, axis=1)

print("\nPolitique apprise sur la grille 4x4 (0 = Start, 15 = Goal):\n")
for i in range(4):
    row = policy[i * 4:(i + 1) * 4]
    print(' '.join([actions[a] for a in row]))
