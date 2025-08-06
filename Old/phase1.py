import gymnasium as gym
import numpy as np
import random
from collections import defaultdict

def flat_mcts(env, num_simulations=10000):
    """
    Flat Monte Carlo Tree Search for one decision step.
    Evaluates each possible action by running multiple random playouts.
    """
    action_scores = defaultdict(list)
    
    for action in range(env.action_space.n):
        for _ in range(num_simulations):
            total_reward = 0
            env_copy = gym.make(env.spec.id)
            obs, _ = env_copy.reset(seed=random.randint(0, 1000))
            
            # try to mimic state for FrozenLake
            try:
                env_copy.env.s = env.env.s
            except AttributeError:
                pass

            obs, reward, terminated, truncated, _ = env_copy.step(action)
            total_reward += reward

            # Perform random rollout
            while not (terminated or truncated):
                next_action = env_copy.action_space.sample()
                obs, reward, terminated, truncated, _ = env_copy.step(next_action)
                total_reward += reward

            action_scores[action].append(total_reward)
        env_copy.close()

    # Choose the action with the highest average reward
    mean_scores = {a: np.mean(scores) for a, scores in action_scores.items()}
    best_action = max(mean_scores, key=mean_scores.get)
    return best_action, mean_scores

def run_flat_mcts_comparison(env_names, episodes=5, num_simulations=100):
    results = {}
    for env_name in env_names:
        print(f"\n🔍 Running Flat MCTS on {env_name}")
        env = gym.make(env_name, render_mode="human")
        total_rewards = []

        for ep in range(episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

            while not (terminated or truncated):
                best_action, _ = flat_mcts(env, num_simulations=num_simulations)
                obs, reward, terminated, truncated, _ = env.step(best_action)
                total_reward += reward

            print(f"Episode {ep+1}: Reward = {total_reward}")
            total_rewards.append(total_reward)

        env.close()
        avg_reward = np.mean(total_rewards)
        results[env_name] = avg_reward
        print(f"🏁 Average reward for {env_name}: {avg_reward:.2f}")
    return results

# Run phase 1 comparison on 3 environments
env_list = ["FrozenLake-v1", "CartPole-v1", "LunarLander-v3"]
run_flat_mcts_comparison(env_list, episodes=3, num_simulations=100)