import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from typing import Dict

class FlatMCTS:
    """
    Flat Monte Carlo Tree Search - Évalue chaque action depuis la racine
    avec des playouts aléatoires
    """
    def __init__(self, env, n_simulations: int = 300, max_depth: int = 100):
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
        
    def random_playout(self, state, action):
        """
        Effectue un playout avec une politique légèrement biaisée
        """
        try:
            env_copy = gym.make(self.env.spec.id)
            env_copy.reset()
            
            # Restaurer l'état
            if hasattr(env_copy.unwrapped, 's'):
                env_copy.unwrapped.s = state
            
            # Première action
            obs, reward, terminated, truncated, _ = env_copy.step(action)
            total_reward = reward
            
            if terminated or truncated:
                env_copy.close()
                return total_reward
            
            # Playout avec politique biaisée
            steps = 0
            current_state = env_copy.unwrapped.s if hasattr(env_copy.unwrapped, 's') else None
            
            while not (terminated or truncated) and steps < self.max_depth:
                # Politique biaisée pour CliffWalking
                if current_state is not None:
                    # Décodage de l'état pour CliffWalking
                    height, width = 4, 12
                    y, x = current_state // width, current_state % width
                    
                    # Biais simple: préférer aller à droite et en haut
                    if np.random.random() < 0.7:  # 70% du temps, utiliser une heuristique simple
                        if y == 3 and x > 0 and x < 11:
                            # Si on est sur la falaise, toujours remonter
                            random_action = 0  # Haut
                        elif y == 2:
                            # Sur la ligne juste au-dessus de la falaise, préférer aller à droite
                            random_action = np.random.choice([0, 2, 3], p=[0.2, 0.7, 0.1])  # Haut, Droite, Gauche
                        else:
                            # Ailleurs, préférer aller en bas ou à droite
                            random_action = np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.4, 0.1])
        
                    else:
                        # 30% du temps, action complètement aléatoire
                        random_action = env_copy.action_space.sample()
                else:
                    # Pour les autres environnements, action aléatoire
                    random_action = env_copy.action_space.sample()
                    
                obs, reward, terminated, truncated, _ = env_copy.step(random_action)
                total_reward += reward
                steps += 1
                
                # Mettre à jour l'état courant
                if hasattr(env_copy.unwrapped, 's'):
                    current_state = env_copy.unwrapped.s
            
            env_copy.close()
            return total_reward
            
        except Exception as e:
            print(f"Erreur dans random_playout: {e}")
            return -100.0  # Valeur par défaut négative pour CliffWalking

        
    def select_action(self, state):
        """
        Sélectionne la meilleure action basée sur les simulations MCTS
        """
        start_time = time.time()
        
        # Initialiser les statistiques pour chaque action
        action_values = np.zeros(self.action_space_size)
        action_counts = np.zeros(self.action_space_size)
        action_stds = np.zeros(self.action_space_size)
        
        # Effectuer des simulations pour chaque action
        for action in range(self.action_space_size):
            rewards = []
            for _ in range(self.n_simulations // self.action_space_size):
                reward = self.random_playout(state, action)
                rewards.append(reward)
                action_values[action] += reward
                action_counts[action] += 1
            
            if action_counts[action] > 0:
                action_values[action] /= action_counts[action]
                action_stds[action] = np.std(rewards) if rewards else 0
        
        # Sélectionner l'action avec la valeur la plus élevée
        best_action = np.argmax(action_values)
        
        # Convertir en dictionnaires pour compatibilité avec le reste du code
        action_values_dict = {a: action_values[a] for a in range(self.action_space_size)}
        action_stds_dict = {a: action_stds[a] for a in range(self.action_space_size)}
        
        computation_time = time.time() - start_time
        
        return best_action, action_values_dict, action_stds_dict

class MCTSEvaluator:
    """
    Classe pour évaluer les performances du Flat MCTS
    """
    def __init__(self):
        self.results = defaultdict(list)
        
    def visualize_cliff_walking_policy(self, agent):
        """Visualise la politique pour CliffWalking-v0"""
        env = gym.make("CliffWalking-v0")
        
        print("\nPolitique apprise pour CliffWalking-v0:")
        
        # Actions: 0=Haut, 1=Bas, 2=Droite, 3=Gauche
        action_symbols = ['↑', '↓', '→', '←']
        
        # Dimensions de la grille
        height, width = 4, 12
        
        # Calculer la politique pour chaque état
        policy = np.zeros(height * width, dtype=int)
        values = np.zeros(height * width)
        
        for state in range(height * width):
            action, action_values, _ = agent.select_action(state)
            policy[state] = action
            values[state] = action_values[action]
        
        # Afficher la politique
        print("\nPolitique (actions):")
        for y in range(height):
            line = ""
            for x in range(width):
                state = y * width + x
                if state == 36:  # État initial (3,0)
                    line += "S"
                elif state == 47:  # But (3,11)
                    line += "G"
                elif y == 3 and x > 0 and x < 11:  # Falaise
                    line += "C"
                else:
                    line += action_symbols[policy[state]]
                line += "\t"
            print(line)
        
        # Afficher les valeurs
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
        """
        Évalue MCTS sur CliffWalking
        """
        print(f"\n=== Évaluation sur {env_name} ===")
        
        env = gym.make(env_name, render_mode="human" if render else None)        
        mcts = FlatMCTS(env, n_simulations=n_simulations, max_depth=100)
        
        scores = []
        success_rate = 0
        episode_lengths = []
        computation_times = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_computation_times = []
            
            while True:
                start_time = time.time()
                
                # Obtenir l'état actuel pour MCTS
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
                
                # Limiter la longueur des épisodes
                if steps >= 50 or terminated or truncated:
                    break
            
            scores.append(total_reward)
            episode_lengths.append(steps)
            computation_times.extend(episode_computation_times)
            
            # Critère de succès pour CliffWalking
            success_rate += (total_reward > -50)  # Succès si moins de 30 pas
            
            if (episode + 1) % 5 == 0:
                print(f"Épisode {episode + 1}/{n_episodes} - Score moyen: {np.mean(scores[-5:]):.2f}")
        
        # Visualiser la politique apprise
        self.visualize_cliff_walking_policy(mcts)
        
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
        fig.suptitle('Flat MCTS - Résultats sur CliffWalking', fontsize=16)
        
        # 1. Scores par épisode
        ax1 = axes[0, 0]
        for env_name, results in self.results.items():
            episodes = range(1, len(results['scores']) + 1)
            ax1.plot(episodes, results['scores'], label=env_name, alpha=0.7)
            # Moyenne mobile
            window = min(5, len(results['scores']) // 5)
            if window > 1:
                moving_avg = np.convolve(results['scores'], 
                                       np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(results['scores']) + 1), 
                        moving_avg, '--', linewidth=2, label='Moyenne mobile')
        
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Score')
        ax1.set_title('Évolution des scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution des scores
        ax2 = axes[0, 1]
        for env_name, results in self.results.items():
            ax2.hist(results['scores'], alpha=0.7, bins=10, label=env_name)
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Longueur des épisodes
        ax3 = axes[1, 0]
        for env_name, results in self.results.items():
            episodes = range(1, len(results['episode_lengths']) + 1)
            ax3.plot(episodes, results['episode_lengths'], label=env_name, alpha=0.7)
            # Moyenne mobile
            window = min(5, len(results['episode_lengths']) // 5)
            if window > 1:
                moving_avg = np.convolve(results['episode_lengths'], 
                                       np.ones(window)/window, mode='valid')
                ax3.plot(range(window, len(results['episode_lengths']) + 1), 
                        moving_avg, '--', linewidth=2, label='Moyenne mobile')
        
        ax3.set_xlabel('Épisode')
        ax3.set_ylabel('Nombre de pas')
        ax3.set_title('Longueur des épisodes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Temps de calcul par épisode
        ax4 = axes[1, 1]
        computation_times = [self.results[env]['avg_computation_time'] for env in self.results]
        bars = ax4.bar(list(self.results.keys()), computation_times, alpha=0.7, color='green')
        ax4.set_xlabel('Environnement')
        ax4.set_ylabel('Temps de calcul (s)')
        ax4.set_title('Temps de calcul moyen par action')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, computation_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('Plots/flat_mcts_cliffwalking_results.png', dpi=300)


def debug_mcts():
    """Test simplifié pour déboguer l'implémentation MCTS sur CliffWalking"""
    env = gym.make("CliffWalking-v0")
    agent = FlatMCTS(env, n_simulations=100, max_depth=30)
    
    # Tester sur quelques épisodes seulement
    for episode in range(2):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nÉpisode {episode+1}")
        print(env.render())
        
        while not done and steps < 30:
            if hasattr(env.unwrapped, 's'):
                current_state = env.unwrapped.s
            else:
                current_state = state
            
            # Décodage de l'état pour CliffWalking
            height, width = 4, 12
            y, x = current_state // width, current_state % width
            print(f"État: ({y},{x}), position {current_state}")
            
            action, values, _ = agent.select_action(current_state)
            action_symbols = ['↑', '↓', '→', '←']
            print(f"Action: {action_symbols[action]} (action {action})")
            print(f"Valeurs: {[f'{a}:{v:.2f}' for a, v in values.items()]}")
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            print(f"Récompense: {reward}, Total: {total_reward}")
            print(env.render())
            print("---")
        
        print(f"Récompense totale: {total_reward}, Étapes: {steps}")
    
    env.close()


def main():
    """
    Fonction principale pour tester le Flat MCTS sur CliffWalking
    """
    # Décommenter pour déboguer
    # debug_mcts()
    
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
        print(f"  • Longueur moyenne des épisodes: {results['avg_length']:.1f}")


if __name__ == "__main__":
    main()
