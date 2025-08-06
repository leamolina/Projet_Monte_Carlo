import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

def clone_env(env):
    """
    Clone un environnement Gymnasium compatible avec CliffWalking.
    Retourne une copie indépendante avec le même état.
    """
    env_copy = gym.make(env.spec.id)
    env_copy.reset()
    # Copier l'état interne si possible
    if hasattr(env.unwrapped, 's'):
        env_copy.unwrapped.s = env.unwrapped.s
    return env_copy

class FlatMCTS:
    def __init__(self, env, n_simulations: int = 300, max_depth: int = 100):
        self.env = env
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.action_space_size = env.action_space.n
        
        # Pré-créer un environnement cloné pour les playouts
        self.env_for_playout = clone_env(env)
        
    def random_playout(self, state, action):
        try:
            env_copy = clone_env(self.env_for_playout)
            # Restaurer l'état
            if hasattr(env_copy.unwrapped, 's'):
                env_copy.unwrapped.s = state
            
            obs, reward, terminated, truncated, _ = env_copy.step(action)
            total_reward = reward
            
            if terminated or truncated:
                env_copy.close()
                return total_reward
            
            steps = 0
            current_state = env_copy.unwrapped.s if hasattr(env_copy.unwrapped, 's') else None
            
            while not (terminated or truncated) and steps < self.max_depth:
                if current_state is not None:
                    height, width = 4, 12
                    y, x = current_state // width, current_state % width
                    
                    if np.random.random() < 0.7:
                        if y == 3 and 0 < x < 11:
                            # Sur la falaise, préférer remonter mais avec un peu de stochasticité
                            random_action = np.random.choice([0, 2], p=[0.8, 0.2])  # Haut ou Droite
                        elif y == 2:
                            random_action = np.random.choice([0, 2, 3], p=[0.2, 0.7, 0.1])
                        else:
                            random_action = np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.4, 0.1])
                    else:
                        random_action = env_copy.action_space.sample()
                else:
                    random_action = env_copy.action_space.sample()
                
                obs, reward, terminated, truncated, _ = env_copy.step(random_action)
                total_reward += reward
                steps += 1
                
                if hasattr(env_copy.unwrapped, 's'):
                    current_state = env_copy.unwrapped.s
            
            env_copy.close()
            return total_reward
            
        except Exception as e:
            print(f"Erreur dans random_playout: {e}")
            return -100.0

    def select_action(self, state):
        start_time = time.time()
        
        action_values = np.zeros(self.action_space_size)
        action_counts = np.zeros(self.action_space_size)
        action_stds = np.zeros(self.action_space_size)
        
        for action in range(self.action_space_size):
            rewards = []
            n_sim = self.n_simulations // self.action_space_size
            for _ in range(n_sim):
                reward = self.random_playout(state, action)
                rewards.append(reward)
            action_values[action] = np.mean(rewards)
            action_counts[action] = n_sim
            action_stds[action] = np.std(rewards)
        
        best_action = np.argmax(action_values)
        
        action_values_dict = {a: action_values[a] for a in range(self.action_space_size)}
        action_stds_dict = {a: action_stds[a] for a in range(self.action_space_size)}
        
        computation_time = time.time() - start_time
        
        return best_action, action_values_dict, action_stds_dict


class MCTSEvaluator:
    def __init__(self):
        self.results = defaultdict(list)
        
    def visualize_cliff_walking_policy(self, agent):
        env = gym.make("CliffWalking-v0")
        
        print("\nPolitique apprise pour CliffWalking-v0:")
        action_symbols = ['↑', '↓', '→', '←']
        height, width = 4, 12
        
        policy = np.zeros(height * width, dtype=int)
        values = np.zeros(height * width)
        
        for state in range(height * width):
            action, action_values, _ = agent.select_action(state)
            policy[state] = action
            values[state] = action_values[action]
        
        print("\nPolitique (actions):")
        for y in range(height):
            line = ""
            for x in range(width):
                state = y * width + x
                if state == 36:
                    line += "S"
                elif state == 47:
                    line += "G"
                elif y == 3 and 0 < x < 11:
                    line += "C"
                else:
                    line += action_symbols[policy[state]]
                line += "\t"
            print(line)
        
        print("\nValeurs des états:")
        for y in range(height):
            line = ""
            for x in range(width):
                state = y * width + x
                line += f"{values[state]:.1f}\t"
            print(line)
        
        env.close()
        
    def evaluate_environment(self, env_name: str, n_episodes: int = 25, 
                             n_simulations: int = 300, render: bool = False):        
        print(f"\n=== Évaluation sur {env_name} ===")
        
        env = gym.make(env_name, render_mode="human" if render else None)        
        mcts = FlatMCTS(env, n_simulations=n_simulations, max_depth=100)
        
        scores = []
        success_count = 0
        episode_lengths = []
        computation_times = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_computation_times = []
            
            while True:
                start_time = time.time()
                
                if hasattr(env.unwrapped, 's'):
                    current_state = env.unwrapped.s
                else:
                    current_state = state
                
                action, action_values, action_stds = mcts.select_action(current_state)
                computation_time = time.time() - start_time
                episode_computation_times.append(computation_time)

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if steps >= 50 or terminated or truncated:
                    break
            
            scores.append(total_reward)
            episode_lengths.append(steps)
            computation_times.extend(episode_computation_times)
            
            # Succès si l'agent atteint le but en moins de 30 pas
            if terminated and steps < 30:
                success_count += 1
            
            if (episode + 1) % 5 == 0:
                print(f"Épisode {episode + 1}/{n_episodes} - Score moyen: {np.mean(scores[-5:]):.2f}")
        
        self.visualize_cliff_walking_policy(mcts)
        
        env.close()
        
        success_rate = success_count / n_episodes * 100
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_length = np.mean(episode_lengths)
        avg_computation_time = np.mean(computation_times)
        
        results = {
            'env_name': env_name,
            'scores': scores,
            'success_rate': success_rate,
            'avg_score': avg_score,
            'std_score': std_score,
            'avg_length': avg_length,
            'avg_computation_time': avg_computation_time,
            'episode_lengths': episode_lengths,
            'n_simulations': n_simulations
        }
        
        self.results[env_name] = results
        
        print(f"Résultats pour {env_name}:")
        print(f"  Score moyen: {avg_score:.2f} ± {std_score:.2f}")
        print(f"  Taux de succès: {success_rate:.1f}%")
        print(f"  Longueur moyenne des épisodes: {avg_length:.1f}")
        print(f"  Temps de calcul moyen par action: {avg_computation_time:.3f}s")
        
        return results
    
    def plot_results(self):
        if not self.results:
            print("Aucun résultat à afficher")
            return
        
        # Création du dossier Plots si nécessaire
        os.makedirs('Plots', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flat MCTS - Résultats sur CliffWalking', fontsize=16)
        
        # Scores par épisode
        ax1 = axes[0, 0]
        for env_name, results in self.results.items():
            episodes = range(1, len(results['scores']) + 1)
            ax1.plot(episodes, results['scores'], label=env_name, alpha=0.7)
            window = min(5, len(results['scores']) // 5)
            if window > 1:
                moving_avg = np.convolve(results['scores'], np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(results['scores']) + 1), moving_avg, '--', linewidth=2, label='Moyenne mobile')
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Score')
        ax1.set_title('Évolution des scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution des scores
        ax2 = axes[0, 1]
        for env_name, results in self.results.items():
            ax2.hist(results['scores'], alpha=0.7, bins=10, label=env_name)
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Longueur des épisodes
        ax3 = axes[1, 0]
        for env_name, results in self.results.items():
            episodes = range(1, len(results['episode_lengths']) + 1)
            ax3.plot(episodes, results['episode_lengths'], label=env_name, alpha=0.7)
            window = min(5, len(results['episode_lengths']) // 5)
            if window > 1:
                moving_avg = np.convolve(results['episode_lengths'], np.ones(window)/window, mode='valid')
                ax3.plot(range(window, len(results['episode_lengths']) + 1), moving_avg, '--', linewidth=2, label='Moyenne mobile')
        ax3.set_xlabel('Épisode')
        ax3.set_ylabel('Nombre de pas')
        ax3.set_title('Longueur des épisodes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Temps de calcul par épisode
        ax4 = axes[1, 1]
        computation_times = [self.results[env]['avg_computation_time'] for env in self.results]
        bars = ax4.bar(list(self.results.keys()), computation_times, alpha=0.7, color='green')
        ax4.set_xlabel('Environnement')
        ax4.set_ylabel('Temps de calcul (s)')
        ax4.set_title('Temps de calcul moyen par action')
        for bar, value in zip(bars, computation_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{value:.3f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('Plots/Chat/flat_mcts_cliffwalking_results(phase1).png', dpi=300)


def main():
    evaluator = MCTSEvaluator()
    
    print("=== Phase 1: Flat Monte Carlo Tree Search sur CliffWalking ===")
    
    try:
        results = evaluator.evaluate_environment(
            env_name="CliffWalking-v0",
            n_episodes=20,
            n_simulations=1000,
            render=False
        )
    except Exception as e:
        print(f"Erreur lors de l'évaluation: {e}")
    
    evaluator.plot_results()
    
    print("\n=== RÉSUMÉ PHASE 1 ===")
    for env_name, results in evaluator.results.items():
        print(f"\n{env_name}:")
        print(f"  • Score moyen: {results['avg_score']:.2f} ± {results['std_score']:.2f}")
        print(f"  • Taux de succès: {results['success_rate']:.1f}%")
        print(f"  • Temps calcul/action: {results['avg_computation_time']:.3f}s")
        print(f"  • Simulations par action: {results['n_simulations']}")
        print(f"  • Longueur moyenne des épisodes: {results['avg_length']:.1f}")


if __name__ == "__main__":
    main()
