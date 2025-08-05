import gymnasium as gym
import numpy as np
import random
import copy

env = gym.make("Breakout-v5", render_mode="human")
num_actions = env.action_space.n  # Usually 4: NOOP, FIRE, RIGHT, LEFT

ROLLOUTS_PER_ACTION = 5
MAX_STEPS = 500

def heuristic_playout(env_copy):
    total_reward = 0
    done = False
    step = 0
    fired = False
    MAX_STEPS = 500

    while not done and step < MAX_STEPS:
        if not fired:
            action = 1  # FIRE
            fired = True
        else:
            action = random.choice([2, 3])  # LEFT or RIGHT
        _, reward, done, truncated, _ = env_copy.step(action)
        total_reward += reward
        if truncated:
            break
        step += 1

    return total_reward


def flat_mcts(env, rollouts_per_action=ROLLOUTS_PER_ACTION):
    legal_actions = list(range(num_actions))
    action_scores = []

    for action in legal_actions:
        scores = []
        for _ in range(rollouts_per_action):
            env_copy = copy.deepcopy(env)
            env_copy.reset()
            env_copy.step(action)
            reward = heuristic_playout(env_copy)
            scores.append(reward)
        avg_score = np.mean(scores)
        action_scores.append(avg_score)

    best_action = legal_actions[np.argmax(action_scores)]
    return best_action

def run_episode():
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = flat_mcts(env)
        _, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if truncated:
            break
    return total_reward

# Run multiple episodes
N_EPISODES = 10
results = []
for ep in range(N_EPISODES):
    score = run_episode()
    print(f"Episode {ep+1}: {score}")
    results.append(score)

print("Average Score:", np.mean(results))
