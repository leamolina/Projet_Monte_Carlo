import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
import pickle

class QLearningAgent:
    """
    Agent Q-Learning tabulaire avec politique ε-greedy
    """
    def __init__(self, n_states: int, n_actions: int, 
             learning_rate: float = 0.1, discount: float = 0.99, 
             epsilon: float = 1.0, epsilon_decay: float = 0.995, 
             epsilon_min: float = 0.01):
        """
        Args:
            n_states: Nombre d'états de l'environnement
            n_actions: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage (α)
            discount: Facteur de discount (γ)
            epsilon: Probabilité d'exploration initiale
            epsilon_decay: Décroissance de epsilon
            epsilon_min: Valeur minimale d'epsilon
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialiser la Q-table
        self.q_table = np.ones((n_states, n_actions)) * 0.1 
        
        # Statistiques d'apprentissage
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.q_table_history = []
        
    def get_action(self, state: int, training: bool = True) -> int:
        """
        Sélectionne une action avec la politique ε-greedy
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool):
        """
        Met à jour la Q-table avec l'équation de Bellman
        """
        if done:
            target = reward
        else:
            target = reward + self.discount * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (
            target - self.q_table[state, action]
        )
    
    def decay_epsilon(self):
        """
        Décroit epsilon après chaque épisode
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_q_table_snapshot(self):
        """
        Sauvegarde un snapshot de la Q-table pour analyse
        """
        self.q_table_history.append(self.q_table.copy())

class StateDiscretizer:
    """
    Discrétise les états continus pour CartPole
    """
    def __init__(self, env_name: str):
        self.env_name = env_name
        
        if env_name == "CartPole-v1":
            # Bornes pour CartPole : [position, velocity, angle, angular_velocity]
            self.bounds = [
                [-2.4, 2.4],    # position du cart
                [-3.0, 3.0],    # vitesse du cart
                [-0.21, 0.21],  # angle du poteau (≈12°)
                [-2.0, 2.0]     # vitesse angulaire
            ]
            self.n_bins = [6, 6, 6, 6]  # Nombre de bins par dimension
            self.n_states = np.prod(self.n_bins)
        
    def discretize_state(self, state):
        """
        Convertit un état continu en état discret
        """
        if self.env_name == "FrozenLake-v1":
            return int(state) if np.isscalar(state) else state
        
        elif self.env_name == "CartPole-v1":
            discrete_state = []
            for i, (value, (low, high), n_bin) in enumerate(
                zip(state, self.bounds, self.n_bins)
            ):
                # Clip la valeur dans les bornes
                value = np.clip(value, low, high)
                # Discrétiser
                bin_idx = int((value - low) / (high - low) * (n_bin - 1))
                bin_idx = min(bin_idx, n_bin - 1)
                discrete_state.append(bin_idx)
            
            # Convertir en index unique
            state_idx = 0
            for i, (bin_idx, n_bin) in enumerate(zip(discrete_state, self.n_bins)):
                state_idx = state_idx * n_bin + bin_idx
            
            return state_idx
        
        return state
def get_shaped_reward(state, next_state, reward, done, env_name):
        """Ajoute une récompense de shaping pour guider l'apprentissage"""
        if env_name != "FrozenLake-v1":
            return reward
            
        if done and reward == 0:  # Punition pour tomber dans un trou
            return -1.0
        
        # Récompense de proximité (Manhattan distance au but)
        goal_pos = (3, 3)  # Position du but dans FrozenLake 4x4
        
        # Convertir l'état en coordonnées 2D
        state_y, state_x = state // 4, state % 4
        next_y, next_x = next_state // 4, next_state % 4
        
        # Distance au but
        prev_dist = abs(state_y - goal_pos[0]) + abs(state_x - goal_pos[1])
        new_dist = abs(next_y - goal_pos[0]) + abs(next_x - goal_pos[1])
        
        # Récompense pour se rapprocher du but
        proximity_reward = 0.01 * (prev_dist - new_dist)
        
        return reward + proximity_reward

class QLearningEvaluator:
    """
    Classe pour évaluer et analyser les performances du Q-Learning
    """

    


    def __init__(self):
        self.results = {}
    
    def train_and_evaluate(self, env_name: str, n_training_episodes: int = 2000,
                          n_evaluation_episodes: int = 100, 
                          hyperparams: Optional[Dict] = None,
                          env_kwargs: Optional[Dict] = None):  # Ajout de env_kwargs
        """
        Entraîne et évalue un agent Q-Learning
        """
        print(f"\n=== Q-Learning sur {env_name} ===")
        
        # Paramètres par défaut
        default_params = {
            'learning_rate': 0.1,
            'discount': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        # Créer l'environnement
        if env_kwargs is None:
            env_kwargs = {}
        env = gym.make(env_name, **env_kwargs)
        
        # Obtenir les dimensions de l'espace d'état
        if env_name == "FrozenLake-v1":
            n_states = env.observation_space.n
            discretizer = StateDiscretizer(env_name)
        elif env_name == "CartPole-v1":
            discretizer = StateDiscretizer(env_name)
            n_states = discretizer.n_states
        else:
            print(f"Environnement {env_name} non supporté pour Q-Learning tabulaire")
            return None
        
        n_actions = env.action_space.n
        
        # Créer l'agent
        agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            **default_params
        )
        
        print(f"Espace d'état: {n_states}, Actions: {n_actions}")
        print(f"Paramètres: {default_params}")
        
        # Phase d'entraînement
        print(f"Entraînement sur {n_training_episodes} épisodes...")
        training_start = time.time()
        
        for episode in range(n_training_episodes):
            state, _ = env.reset()
            state = discretizer.discretize_state(state)
            
            total_reward = 0
            steps = 0
            
            while True:
                action = agent.get_action(state, training=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = discretizer.discretize_state(next_obs)
                
                # Appliquer le reward shaping
                shaped_reward = get_shaped_reward(state, next_state, reward, terminated or truncated, env_name)

                agent.update_q_table(state, action, shaped_reward, next_state, terminated or truncated)
                                
                state = next_state
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            agent.episode_rewards.append(total_reward)
            agent.episode_lengths.append(steps)
            agent.epsilon_history.append(agent.epsilon)
            agent.decay_epsilon()
            
            # Sauvegarder des snapshots de la Q-table
            if episode % (n_training_episodes // 10) == 0:
                agent.save_q_table_snapshot()
            
            if (episode + 1) % (n_training_episodes // 5) == 0:
                recent_avg = np.mean(agent.episode_rewards[-100:])
                print(f"Épisode {episode + 1}/{n_training_episodes} - "
                      f"Score moyen (100 derniers): {recent_avg:.2f} - "
                      f"ε: {agent.epsilon:.3f}")
        
        training_time = time.time() - training_start
        
        # Phase d'évaluation
        print(f"Évaluation sur {n_evaluation_episodes} épisodes...")
        eval_scores = []
        eval_lengths = []
        success_count = 0
        
        for episode in range(n_evaluation_episodes):
            state, _ = env.reset()
            state = discretizer.discretize_state(state)
            
            total_reward = 0
            steps = 0
            
            while True:
                action = agent.get_action(state, training=False)  # Pas d'exploration
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = discretizer.discretize_state(next_obs)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            eval_scores.append(total_reward)
            eval_lengths.append(steps)
            
            # Critère de succès
            if env_name == "FrozenLake-v1":
                success_count += (total_reward > 0)
            elif env_name == "CartPole-v1":
                success_count += (total_reward >= 195)
        
        env.close()
        
        # Calculer les statistiques
        results = {
            'env_name': env_name,
            'hyperparams': default_params,
            'n_training_episodes': n_training_episodes,
            'training_time': training_time,
            'training_rewards': agent.episode_rewards,
            'training_lengths': agent.episode_lengths,
            'epsilon_history': agent.epsilon_history,
            'eval_scores': eval_scores,
            'eval_lengths': eval_lengths,
            'eval_avg_score': np.mean(eval_scores),
            'eval_std_score': np.std(eval_scores),
            'success_rate': success_count / n_evaluation_episodes * 100,
            'final_q_table': agent.q_table.copy(),
            'q_table_history': agent.q_table_history,
            'convergence_episode': self._find_convergence_point(agent.episode_rewards)
        }
        
        self.results[env_name] = results
        
        print(f"Résultats pour {env_name}:")
        print(f"  Score d'évaluation: {results['eval_avg_score']:.2f} ± {results['eval_std_score']:.2f}")
        print(f"  Taux de succès: {results['success_rate']:.1f}%")
        print(f"  Temps d'entraînement: {training_time:.1f}s")
        print(f"  Convergence vers épisode: {results['convergence_episode']}")
        
        return results
    
    def _find_convergence_point(self, rewards: List[float], window: int = 100) -> int:
        """
        Trouve approximativement le point de convergence
        """
        if len(rewards) < window * 2:
            return len(rewards)
        
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        # Chercher où la variance devient faible
        for i in range(len(moving_avg) - window):
            recent_variance = np.var(moving_avg[i:i+window])
            if recent_variance < 0.1:  # Seuil arbitraire
                return i + window
        
        return len(rewards)
    
    def plot_training_results(self, env_name: str):
        """
        Affiche les graphiques d'entraînement pour un environnement
        """
        if env_name not in self.results:
            print(f"Pas de résultats pour {env_name}")
            return
        
        results = self.results[env_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Q-Learning - Résultats d\'entraînement ({env_name})', fontsize=16)
        
        episodes = range(1, len(results['training_rewards']) + 1)
        
        # 1. Évolution des rewards
        ax1 = axes[0, 0]
        ax1.plot(episodes, results['training_rewards'], alpha=0.6, linewidth=0.8)
        
        # Moyenne mobile
        window = min(100, len(results['training_rewards']) // 10)
        if window > 1:
            moving_avg = np.convolve(results['training_rewards'], 
                                   np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(results['training_rewards']) + 1), 
                    moving_avg, 'r-', linewidth=2, label=f'Moyenne mobile ({window})')
        
        ax1.axvline(x=results['convergence_episode'], color='g', linestyle='--', 
                   label='Convergence approx.')
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Évolution des rewards d\'entraînement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Évolution d'epsilon
        ax2 = axes[0, 1]
        ax2.plot(episodes, results['epsilon_history'], 'orange', linewidth=2)
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Décroissance d\'epsilon (exploration)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution des rewards d'évaluation
        ax3 = axes[0, 2]
        ax3.hist(results['eval_scores'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=results['eval_avg_score'], color='red', linestyle='--', 
                   label=f'Moyenne: {results["eval_avg_score"]:.2f}')
        ax3.set_xlabel('Score d\'évaluation')
        ax3.set_ylabel('Fréquence')
        ax3.set_title('Distribution des scores d\'évaluation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Longueur des épisodes
        ax4 = axes[1, 0]
        ax4.plot(episodes, results['training_lengths'], alpha=0.6, linewidth=0.8)
        
        if window > 1:
            moving_avg_length = np.convolve(results['training_lengths'], 
                                          np.ones(window)/window, mode='valid')
            ax4.plot(range(window, len(results['training_lengths']) + 1), 
                    moving_avg_length, 'r-', linewidth=2, label=f'Moyenne mobile ({window})')
        
        ax4.set_xlabel('Épisode')
        ax4.set_ylabel('Nombre de steps')
        ax4.set_title('Longueur des épisodes d\'entraînement')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Heatmap de la Q-table finale
        ax5 = axes[1, 1]
        if env_name == "FrozenLake-v1":
            # Afficher la Q-table sous forme de heatmap
            q_table = results['final_q_table']
            im = ax5.imshow(q_table.T, cmap='viridis', aspect='auto')
            ax5.set_xlabel('États')
            ax5.set_ylabel('Actions')
            ax5.set_title('Q-table finale (Heatmap)')
            plt.colorbar(im, ax=ax5)
        else:
            # Pour CartPole, afficher les valeurs moyennes par action
            q_table = results['final_q_table']
            action_values = np.mean(q_table, axis=0)
            bars = ax5.bar(range(len(action_values)), action_values, 
                          color=['blue', 'orange'][:len(action_values)])
            ax5.set_xlabel('Actions')
            ax5.set_ylabel('Valeur Q moyenne')
            ax5.set_title('Valeurs Q moyennes par action')
            ax5.set_xticks(range(len(action_values)))
            
            # Ajouter les valeurs sur les barres
            for bar, value in zip(bars, action_values):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # 6. Évolution de la Q-table (variance)
        ax6 = axes[1, 2]
        if results['q_table_history']:
            q_variances = []
            for q_snapshot in results['q_table_history']:
                q_variances.append(np.var(q_snapshot))
            
            snapshot_episodes = np.linspace(0, len(results['training_rewards']), 
                                          len(q_variances))
            ax6.plot(snapshot_episodes, q_variances, 'g-', linewidth=2, marker='o')
            ax6.set_xlabel('Épisode')
            ax6.set_ylabel('Variance de la Q-table')
            ax6.set_title('Stabilisation de la Q-table')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Pas de snapshots\nde Q-table', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Évolution Q-table non disponible')
        
        plt.tight_layout()
        plt.show()
    
    def compare_hyperparameters(self, env_name: str, param_grid: Dict):
        """
        Compare différentes combinaisons d'hyperparamètres
        """
        print(f"\n=== Comparaison d'hyperparamètres pour {env_name} ===")
        
        results_comparison = {}
        
        # Tester différentes combinaisons
        for lr in param_grid.get('learning_rates', [0.1]):
            for discount in param_grid.get('discounts', [0.99]):
                for epsilon_decay in param_grid.get('epsilon_decays', [0.995]):
                    
                    params = {
                        'learning_rate': lr,
                        'discount': discount,
                        'epsilon_decay': epsilon_decay
                    }
                    
                    key = f"lr={lr}_γ={discount}_εd={epsilon_decay}"
                    print(f"\nTest: {key}")
                    
                    # Entraînement plus court pour la comparaison
                    result = self.train_and_evaluate(
                        env_name=env_name, 
                        n_training_episodes=1000,
                        n_evaluation_episodes=50,
                        hyperparams=params
                    )
                    
                    if result:
                        results_comparison[key] = result
        
        # Afficher la comparaison
        self._plot_hyperparameter_comparison(results_comparison, env_name)
        
        return results_comparison
    
    def _plot_hyperparameter_comparison(self, results_dict: Dict, env_name: str):
        """
        Affiche la comparaison des hyperparamètres
        """
        if not results_dict:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Comparaison d\'hyperparamètres - {env_name}', fontsize=16)
        
        param_names = list(results_dict.keys())
        
        # 1. Score d'évaluation moyen
        ax1 = axes[0, 0]
        eval_scores = [results_dict[name]['eval_avg_score'] for name in param_names]
        bars = ax1.bar(range(len(param_names)), eval_scores, alpha=0.7)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Score moyen')
        ax1.set_title('Score d\'évaluation moyen')
        ax1.set_xticks(range(len(param_names)))
        ax1.set_xticklabels(param_names, rotation=45, ha='right')
        
        # Ajouter les valeurs
        for bar, score in zip(bars, eval_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 2. Taux de succès
        ax2 = axes[0, 1]
        success_rates = [results_dict[name]['success_rate'] for name in param_names]
        bars = ax2.bar(range(len(param_names)), success_rates, alpha=0.7, color='green')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Taux de succès (%)')
        ax2.set_title('Taux de succès')
        ax2.set_xticks(range(len(param_names)))
        ax2.set_xticklabels(param_names, rotation=45, ha='right')
        
        for bar, rate in zip(bars, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Convergence
        ax3 = axes[1, 0]
        convergence_points = [results_dict[name]['convergence_episode'] for name in param_names]
        bars = ax3.bar(range(len(param_names)), convergence_points, alpha=0.7, color='orange')
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Épisode de convergence')
        ax3.set_title('Vitesse de convergence')
        ax3.set_xticks(range(len(param_names)))
        ax3.set_xticklabels(param_names, rotation=45, ha='right')
        
        for bar, conv in zip(bars, convergence_points):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{conv}', ha='center', va='bottom')
        
        # 4. Courbes d'apprentissage
        ax4 = axes[1, 1]
        for name, results in results_dict.items():
            # Calculer la moyenne mobile pour chaque configuration
            rewards = results['training_rewards']
            window = min(50, len(rewards) // 10)
            if window > 1:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                episodes = range(window, len(rewards) + 1)
                ax4.plot(episodes, moving_avg, label=name, alpha=0.8)
        
        ax4.set_xlabel('Épisode')
        ax4.set_ylabel('Reward (moyenne mobile)')
        ax4.set_title('Courbes d\'apprentissage')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Fonction principale pour tester le Q-Learning
    """
    evaluator = QLearningEvaluator()
    
    print("=== Phase 2: Q-Learning Tabulaire ===")
    print("Entraînement et évaluation d'agents Q-Learning\n")
    
    # Test sur FrozenLake slippery=True
    print("1. Test sur FrozenLake-v1 (slippery=True)")
    frozen_results_slippery = evaluator.train_and_evaluate(
    env_name="FrozenLake-v1",
    n_training_episodes=5000,  # Plus d'épisodes
    n_evaluation_episodes=100,
    hyperparams={
        'learning_rate': 0.1,
        'discount': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.9995,  # Décroissance plus lente
        'epsilon_min': 0.1  # Minimum plus élevé
    },
    env_kwargs={'is_slippery': True}
)
    if frozen_results_slippery:
        evaluator.plot_training_results("FrozenLake-v1")
    
    # Test sur FrozenLake slippery=False
    print("\n2. Test sur FrozenLake-v1 (slippery=False)")
    frozen_results_nonslippery = evaluator.train_and_evaluate(
    env_name="FrozenLake-v1",
    n_training_episodes=5000,  # Plus d'épisodes
    n_evaluation_episodes=100,
    hyperparams={
        'learning_rate': 0.1,
        'discount': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.9995,  # Décroissance plus lente
        'epsilon_min': 0.1  # Minimum plus élevé
    },
    env_kwargs={'is_slippery': True}
)
    if frozen_results_nonslippery:
        evaluator.plot_training_results("FrozenLake-v1")
    
    # Comparaison d'hyperparamètres sur FrozenLake slippery=True
    print("\n4. Comparaison d'hyperparamètres sur FrozenLake (slippery=True)")
    param_grid = {
        'learning_rates': [0.05, 0.1, 0.2],
        'discounts': [0.95, 0.99],
        'epsilon_decays': [0.99, 0.995]
    }
    comparison_results = evaluator.compare_hyperparameters("FrozenLake-v1", param_grid)
    
    # Résumé final
    print("\n=== RÉSUMÉ PHASE 2 ===")
    for env_name, results in evaluator.results.items():
        if 'eval_avg_score' in results:  # Éviter les doublons de la comparaison
            print(f"\n{env_name}:")
            print(f"  • Score d'évaluation: {results['eval_avg_score']:.2f} ± {results['eval_std_score']:.2f}")
            print(f"  • Taux de succès: {results['success_rate']:.1f}%")
            print(f"  • Convergence: ~{results['convergence_episode']} épisodes")
            print(f"  • Temps d'entraînement: {results['training_time']:.1f}s")

if __name__ == "__main__":
    main()