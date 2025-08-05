import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
from collections import defaultdict

class FlatMCTS:
    """
    Flat Monte Carlo Tree Search - Évalue chaque action depuis la racine
    avec des playouts aléatoires
    """
    def __init__(self, env, n_simulations: int = 1000, max_depth: int = 100):
        """
        Args:
            env: Environnement Gymnasium
            n_simulations: Nombre de simulations par action
            max_depth: Profondeur maximale des playouts
        """
        self.env = env
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.action_space_size = env.action_space.n

    @staticmethod
    def epsilon_greedy_action(env, epsilon=0.2):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            # Heuristique simple : toujours aller à droite ou en bas (actions 1 ou 2)
            return np.random.choice([1, 2])

        
    def random_playout(self, state, action):
        """
        Effectue un playout aléatoire depuis l'état donné avec l'action initiale
        """
        # Créer une copie de l'environnement pour la simulation
        env_copy = gym.make(self.env.spec.id)
        env_copy.reset()
        
        # Restaurer l'état (approximatif pour certains environnements)
        if hasattr(env_copy, 'unwrapped'):
            if hasattr(env_copy.unwrapped, 's'):  # FrozenLake
                env_copy.unwrapped.s = state
            elif hasattr(env_copy.unwrapped, 'state'):  # CartPole, LunarLander
                env_copy.unwrapped.state = state
        
        # Première action
        obs, reward, terminated, truncated, _ = env_copy.step(action)
        total_reward = reward
        
        # Playout aléatoire
        steps = 0
        while not (terminated or truncated) and steps < self.max_depth:
            random_action = FlatMCTS.epsilon_greedy_action(env_copy)
            obs, reward, terminated, truncated, _ = env_copy.step(random_action)
            total_reward += reward
            steps += 1
            
        env_copy.close()
        return total_reward
    
    def evaluate_action(self, state, action):
        """
        Évalue une action en effectuant plusieurs playouts
        """
        rewards = []
        for _ in range(self.n_simulations):
            reward = self.random_playout(state, action)
            rewards.append(reward)
        
        return np.mean(rewards), np.std(rewards)
    
    def select_action(self, state):
        """
        Sélectionne la meilleure action basée sur les évaluations MCTS
        """
        action_values = {}
        action_stds = {}
        
        for action in range(self.action_space_size):
            mean_reward, std_reward = self.evaluate_action(state, action)
            action_values[action] = mean_reward
            action_stds[action] = std_reward
        
        # Sélectionner l'action avec la plus haute valeur moyenne
        best_action = max(action_values, key=action_values.get)
        
        return best_action, action_values, action_stds

class MCTSEvaluator:
    """
    Classe pour évaluer les performances du Flat MCTS
    """
    def __init__(self):
        self.results = defaultdict(list)
        
    def evaluate_environment(self, env_name: str, n_episodes: int = 100, 
                         n_simulations: int = 500, render: bool = False,
                         env_kwargs: Dict = None):        
        """
        Évalue MCTS sur un environnement donné
        """
        print(f"\n=== Évaluation sur {env_name} ===")
        
        env_kwargs = env_kwargs or {}
        env = gym.make(env_name, render_mode="human" if render else None, **env_kwargs)        
        mcts = FlatMCTS(env, n_simulations=n_simulations)
        
        scores = []
        success_rate = 0
        episode_lengths = []
        computation_times = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_start = time.time()
            
            while True:
                start_time = time.time()
                
                # Obtenir l'état actuel pour MCTS
                if hasattr(env.unwrapped, 's'):  # FrozenLake
                    current_state = env.unwrapped.s
                elif hasattr(env.unwrapped, 'state'):  # CartPole, LunarLander
                    current_state = env.unwrapped.state
                else:
                    current_state = state
                
                action, action_values, action_stds = mcts.select_action(current_state)
                computation_time = time.time() - start_time
                computation_times.append(computation_time)

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            scores.append(total_reward)
            episode_lengths.append(steps)
            
            # Critère de succès spécifique à l'environnement
            if env_name == "FrozenLake-v1":
                success_rate += (total_reward > 0)
            elif env_name == "CartPole-v1":
                success_rate += (total_reward >= 195)  # Seuil de succès CartPole
            elif env_name == "LunarLander-v2":
                success_rate += (total_reward >= 200)  # Seuil de succès LunarLander
            
            if (episode + 1) % 20 == 0:
                print(f"Épisode {episode + 1}/{n_episodes} - Score moyen: {np.mean(scores[-20:]):.2f}")
        
        env.close()
        
        # Calculer les statistiques
        success_rate = success_rate / n_episodes * 100
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
        """
        Affiche les graphiques des résultats
        """
        n_envs = len(self.results)
        if n_envs == 0:
            print("Aucun résultat à afficher")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flat MCTS - Résultats d\'évaluation', fontsize=16)
        
        # 1. Scores par épisode
        ax1 = axes[0, 0]
        for env_name, results in self.results.items():
            episodes = range(1, len(results['scores']) + 1)
            ax1.plot(episodes, results['scores'], label=env_name, alpha=0.7)
            # Moyenne mobile
            window = min(10, len(results['scores']) // 5)
            if window > 1:
                moving_avg = np.convolve(results['scores'], 
                                       np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(results['scores']) + 1), 
                        moving_avg, '--', linewidth=2)
        
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Score')
        ax1.set_title('Évolution des scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution des scores
        ax2 = axes[0, 1]
        for env_name, results in self.results.items():
            ax2.hist(results['scores'], alpha=0.6, bins=20, label=env_name)
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparaison des métriques
        ax3 = axes[1, 0]
        env_names = list(self.results.keys())
        success_rates = [self.results[env]['success_rate'] for env in env_names]
        avg_scores = [self.results[env]['avg_score'] for env in env_names]
        
        x = np.arange(len(env_names))
        width = 0.35
        
        ax3_twin = ax3.twinx()
        bars1 = ax3.bar(x - width/2, success_rates, width, label='Taux de succès (%)', 
                       color='skyblue', alpha=0.7)
        bars2 = ax3_twin.bar(x + width/2, avg_scores, width, label='Score moyen', 
                            color='lightcoral', alpha=0.7)
        
        ax3.set_xlabel('Environnement')
        ax3.set_ylabel('Taux de succès (%)', color='blue')
        ax3_twin.set_ylabel('Score moyen', color='red')
        ax3.set_title('Comparaison des performances')
        ax3.set_xticks(x)
        ax3.set_xticklabels(env_names, rotation=45)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars1, success_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        for bar, value in zip(bars2, avg_scores):
            ax3_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                         f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Temps de calcul
        ax4 = axes[1, 1]
        computation_times = [self.results[env]['avg_computation_time'] for env in env_names]
        bars = ax4.bar(env_names, computation_times, alpha=0.7, color='green')
        ax4.set_xlabel('Environnement')
        ax4.set_ylabel('Temps de calcul (s)')
        ax4.set_title('Temps de calcul moyen par action')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, computation_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Fonction principale pour tester le Flat MCTS
    """
    evaluator = MCTSEvaluator()
    
    # Environnements recommandés avec paramètres adaptés
    environments = [
    ("FrozenLake-v1", 100, 200, {"is_slippery": True}), # (env_name, n_episodes, n_simulations, env_kwargs)
    ("FrozenLake-v1", 100, 200, {"is_slippery": False}),
    ("CartPole-v1", 50, 100, {}), # Moins de simulations car plus rapide
    ("LunarLander-v2", 30, 150, {}), # Moins d'épisodes car plus lent

    ]
    
    print("=== Phase 1: Flat Monte Carlo Tree Search ===")
    print("Évaluation sur différents environnements Gymnasium\n")
    
    for env_name, n_episodes, n_simulations, env_kwargs in environments:
        label = f"{env_name} (slippery={env_kwargs.get('is_slippery', 'default')})"
        try:
            results = evaluator.evaluate_environment(
                env_name=env_name,
                n_episodes=n_episodes,
                n_simulations=n_simulations,
                render=False,
                env_kwargs=env_kwargs
            )
            # Renommer pour affichage correct
            evaluator.results[label] = evaluator.results.pop(env_name)
        except Exception as e:
            print(f"Erreur avec {label}: {e}")
            continue

    # Afficher les graphiques
    evaluator.plot_results()
    
    # Afficher un résumé
    print("\n=== RÉSUMÉ PHASE 1 ===")
    for env_name, results in evaluator.results.items():
        print(f"\n{env_name}:")
        print(f"  • Score moyen: {results['avg_score']:.2f} ± {results['std_score']:.2f}")
        print(f"  • Taux de succès: {results['success_rate']:.1f}%")
        print(f"  • Temps calcul/action: {results['avg_computation_time']:.3f}s")
        print(f"  • Simulations par action: {results['n_simulations']}")

if __name__ == "__main__":
    main()