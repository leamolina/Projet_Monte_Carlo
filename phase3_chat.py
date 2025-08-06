import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
import os

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, action_space_size):
        return len(self.children) == action_space_size

    def best_child(self, c_param=1.4):
        choices_weights = []
        for action, child in self.children.items():
            exploitation = child.value / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            choices_weights.append((exploitation + exploration, action, child))
        return max(choices_weights, key=lambda x: x[0])[1:]

class MCTS:
    def __init__(self, env, n_simulations=300, max_depth=100, c_param=1.4):
        self.env = env
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.c_param = c_param
        self.action_space_size = env.action_space.n
        self.root = None
        self.env_for_sim = self.clone_env(env)

    def clone_env(self, env):
        env_copy = gym.make(env.spec.id)
        env_copy.reset()
        if hasattr(env.unwrapped, 's'):
            env_copy.unwrapped.s = env.unwrapped.s
        return env_copy

    def playout(self, env, max_depth):
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < max_depth:
            state = env.unwrapped.s if hasattr(env.unwrapped, 's') else None
            if state is not None:
                # Politique biaisée simple pour CliffWalking
                height, width = 4, 12
                y, x = state // width, state % width
                if np.random.random() < 0.7:
                    if y == 3 and 0 < x < 11:
                        action = np.random.choice([0, 2], p=[0.8, 0.2])  # Haut ou Droite
                    elif y == 2:
                        action = np.random.choice([0, 2, 3], p=[0.2, 0.7, 0.1])
                    else:
                        action = np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.4, 0.1])
                else:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

        return total_reward

    def tree_policy(self, node):
        # Sélectionner un noeud feuille à développer
        while True:
            if node.visits == 0 or node.parent is None:
                return node
            if not node.is_fully_expanded(self.action_space_size):
                return node
            # Sinon, choisir le meilleur enfant selon UCT
            _, action, node = node.best_child(self.c_param)
        return node

    def expand(self, node):
        # Étendre un enfant non exploré
        tried_actions = node.children.keys()
        for action in range(self.action_space_size):
            if action not in tried_actions:
                # Créer un nouvel enfant
                new_state = self.simulate_action(node.state, action)
                child_node = MCTSNode(new_state, parent=node)
                node.children[action] = child_node
                return child_node
        # Tous les enfants sont déjà explorés
        return None

    def simulate_action(self, state, action):
        # Simuler l'action sur un environnement cloné pour obtenir le nouvel état
        env_copy = self.clone_env(self.env_for_sim)
        if hasattr(env_copy.unwrapped, 's'):
            env_copy.unwrapped.s = state
        _, _, terminated, truncated, _ = env_copy.step(action)
        new_state = env_copy.unwrapped.s if hasattr(env_copy.unwrapped, 's') else None
        env_copy.close()
        return new_state

    def backpropagate(self, node, reward):
        # Remonter la récompense dans l'arbre
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def best_action(self, state):
        self.root = MCTSNode(state)
        for _ in range(self.n_simulations):
            node = self.tree_policy(self.root)
            if node.visits > 0 and not node.is_fully_expanded(self.action_space_size):
                node = self.expand(node)
            # Simuler un playout à partir de l'état du noeud
            env_sim = self.clone_env(self.env_for_sim)
            if hasattr(env_sim.unwrapped, 's'):
                env_sim.unwrapped.s = node.state
            reward = self.playout(env_sim, self.max_depth)
            env_sim.close()
            self.backpropagate(node, reward)

        # Choisir l'action avec la meilleure moyenne
        best_action = None
        best_value = -float('inf')
        for action, child in self.root.children.items():
            avg_value = child.value / child.visits if child.visits > 0 else -float('inf')
            if avg_value > best_value:
                best_value = avg_value
                best_action = action

        return best_action
class MCTSEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate(self, env_name="CliffWalking-v0", n_episodes=20, n_simulations=500, max_depth=100):
        env = gym.make(env_name)
        mcts = MCTS(env, n_simulations=n_simulations, max_depth=max_depth)
        scores = []
        success_count = 0
        episode_lengths = []
        computation_times = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 100:
                if hasattr(env.unwrapped, 's'):
                    current_state = env.unwrapped.s
                else:
                    current_state = state

                start_time = time.time()
                action = mcts.best_action(current_state)
                elapsed = time.time() - start_time
                computation_times.append(elapsed)

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            scores.append(total_reward)
            episode_lengths.append(steps)
            if terminated and steps < 30:
                success_count += 1

            print(f"Episode {episode+1}: Reward={total_reward}, Steps={steps}")

        env.close()

        success_rate = success_count / n_episodes * 100
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_length = np.mean(episode_lengths)
        avg_computation_time = np.mean(computation_times)

        self.results[env_name] = {
            'scores': scores,
            'success_rate': success_rate,
            'avg_score': avg_score,
            'std_score': std_score,
            'avg_length': avg_length,
            'avg_computation_time': avg_computation_time,
            'episode_lengths': episode_lengths,
            'n_simulations': n_simulations
        }

        print(f"\nRésultats sur {env_name}:")
        print(f"Score moyen: {avg_score:.2f} ± {std_score:.2f}")
        print(f"Taux de succès (arrivée <30 pas): {success_rate:.1f}%")
        print(f"Longueur moyenne des épisodes: {avg_length:.1f}")
        print(f"Temps calcul moyen par action: {avg_computation_time:.3f}s")

    def plot_results(self):
        if not self.results:
            print("Aucun résultat à afficher")
            return

        os.makedirs('Plots', exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MCTS Optimisé - Résultats', fontsize=16)

        # Scores par épisode
        ax1 = axes[0, 0]
        for env_name, res in self.results.items():
            episodes = range(1, len(res['scores']) + 1)
            ax1.plot(episodes, res['scores'], label=env_name, alpha=0.7)
            window = min(5, len(res['scores']) // 5)
            if window > 1:
                moving_avg = np.convolve(res['scores'], np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(res['scores']) + 1), moving_avg, '--', linewidth=2, label='Moyenne mobile')
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Score')
        ax1.set_title('Évolution des scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Distribution des scores
        ax2 = axes[0, 1]
        for env_name, res in self.results.items():
            ax2.hist(res['scores'], alpha=0.7, bins=10, label=env_name)
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Longueur des épisodes
        ax3 = axes[1, 0]
        for env_name, res in self.results.items():
            episodes = range(1, len(res['episode_lengths']) + 1)
            ax3.plot(episodes, res['episode_lengths'], label=env_name, alpha=0.7)
            window = min(5, len(res['episode_lengths']) // 5)
            if window > 1:
                moving_avg = np.convolve(res['episode_lengths'], np.ones(window)/window, mode='valid')
                ax3.plot(range(window, len(res['episode_lengths']) + 1), moving_avg, '--', linewidth=2, label='Moyenne mobile')
        ax3.set_xlabel('Épisode')
        ax3.set_ylabel('Nombre de pas')
        ax3.set_title('Longueur des épisodes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Temps de calcul par action
        ax4 = axes[1, 1]
        computation_times = [res['avg_computation_time'] for res in self.results.values()]
        bars = ax4.bar(list(self.results.keys()), computation_times, alpha=0.7, color='green')
        ax4.set_xlabel('Environnement')
        ax4.set_ylabel('Temps de calcul (s)')
        ax4.set_title('Temps de calcul moyen par action')
        for bar, value in zip(bars, computation_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{value:.3f}s', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('Plots/Chat/mcts_optimized_results(phase3).png', dpi=300)
        plt.show()


def main():
    evaluator = MCTSEvaluator()
    evaluator.evaluate(
        env_name="CliffWalking-v0",
        n_episodes=20,
        n_simulations=500,
        max_depth=100
    )
    evaluator.plot_results()

if __name__ == "__main__":
    main()
