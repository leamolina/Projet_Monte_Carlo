import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
import random

class OptimizedMCTS:
    """
    MCTS Optimisé basé sur l'article "Optimized Monte Carlo Tree Search 
    for Enhanced Decision Making in the FrozenLake Environment"
    
    Améliorations implémentées:
    1. Playout biaisé avec ε-greedy
    2. Estimation de valeur d'état basée sur l'historique
    3. Playout adaptatif basé sur la confiance
    4. Réutilisation partielle des statistiques
    """
    
    def __init__(self, env, n_simulations: int = 1000, max_depth: int = 100,
                 epsilon: float = 0.3, value_bias_weight: float = 0.5,
                 confidence_threshold: float = 0.1, memory_decay: float = 0.95):
        """
        Args:
            env: Environnement Gymnasium
            n_simulations: Nombre de simulations par action
            max_depth: Profondeur maximale des playouts
            epsilon: Probabilité d'action aléatoire dans playout biaisé
            value_bias_weight: Poids du biais de valeur d'état
            confidence_threshold: Seuil de confiance pour playout adaptatif
            memory_decay: Facteur de décroissance pour la mémoire statistique
        """
        self.env = env
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.epsilon = epsilon
        self.value_bias_weight = value_bias_weight
        self.confidence_threshold = confidence_threshold
        self.memory_decay = memory_decay
        self.action_space_size = env.action_space.n
        
        # Mémoire statistique pour l'estimation de valeur d'état
        self.state_values = defaultdict(float)  # V(s) estimé
        self.state_visits = defaultdict(int)    # Nombre de visites
        self.state_rewards = defaultdict(list)  # Historique des rewards
        
        # Statistiques de performance
        self.simulation_counts = []
        self.confidence_scores = []
        self.bias_usage = []
    
    def get_state_key(self, state):
        """
        Convertit un état en clé pour le dictionnaire
        """
        if hasattr(self.env.unwrapped, 's'):  # FrozenLake
            return self.env.unwrapped.s
        elif hasattr(self.env.unwrapped, 'state'):  # CartPole, LunarLander
            # Pour les états continus, on peut les discrétiser ou utiliser une approximation
            if isinstance(state, (list, np.ndarray)):
                return tuple(np.round(state, 2))  # Arrondir pour créer des groupes
            return state
        return state
    
    def update_state_memory(self, state_key, reward):
        """
        Met à jour la mémoire statistique pour un état
        """
        self.state_visits[state_key] += 1
        self.state_rewards[state_key].append(reward)
        
        # Appliquer la décroissance sur les anciens rewards
        if len(self.state_rewards[state_key]) > 10:
            old_rewards = self.state_rewards[state_key][:-10]
            decayed_rewards = [r * self.memory_decay for r in old_rewards]
            self.state_rewards[state_key] = decayed_rewards + self.state_rewards[state_key][-10:]
        
        # Mettre à jour la valeur d'état estimée
        self.state_values[state_key] = np.mean(self.state_rewards[state_key])
    
    def get_state_value_estimate(self, state_key):
        """
        Obtient l'estimation de la valeur d'un état
        """
        if state_key in self.state_values:
            return self.state_values[state_key]
        return 0.0  # Valeur par défaut pour les états non visités
    
    def get_state_confidence(self, state_key):
        """
        Calcule la confiance dans l'estimation d'un état
        """
        if state_key not in self.state_visits or self.state_visits[state_key] < 2:
            return 0.0
        
        rewards = self.state_rewards[state_key]
        if len(rewards) < 2:
            return 0.0
        
        # Confiance basée sur l'inverse de la variance normalisée
        variance = np.var(rewards)
        confidence = 1.0 / (1.0 + variance)
        return confidence
    
    def biased_playout(self, state, first_action):
        """
        Playout biaisé qui utilise une politique ε-greedy informée
        """
        env_copy = gym.make(self.env.spec.id)
        env_copy.reset()
        
        # Restaurer l'état initial
        if hasattr(env_copy.unwrapped, 's'):  # FrozenLake
            env_copy.unwrapped.s = state
        elif hasattr(env_copy.unwrapped, 'state'):  # CartPole, LunarLander
            env_copy.unwrapped.state = state
        
        # Première action imposée
        obs, reward, terminated, truncated, _ = env_copy.step(first_action)
        total_reward = reward
        
        # Mise à jour de la mémoire pour l'état initial
        state_key = self.get_state_key(state)
        
        # Playout avec politique biaisée
        steps = 0
        while not (terminated or truncated) and steps < self.max_depth:
            current_state_key = self.get_state_key(obs)
            
            if random.random() < self.epsilon:
                # Action aléatoire (exploration)
                action = env_copy.action_space.sample()
                self.bias_usage.append(0)  # Pas de biais utilisé
            else:
                # Action biaisée basée sur les valeurs d'état estimées
                best_action = 0
                best_value = float('-inf')
                
                for a in range(self.action_space_size):
                    # Simuler temporairement l'action pour évaluer l'état suivant
                    temp_env = gym.make(self.env.spec.id)
                    temp_env.reset()
                    
                    if hasattr(temp_env.unwrapped, 's'):
                        temp_env.unwrapped.s = current_state_key if isinstance(current_state_key, int) else 0
                    
                    try:
                        next_obs, _, term, trunc, _ = temp_env.step(a)
                        if not (term or trunc):
                            next_state_key = self.get_state_key(next_obs)
                            state_value = self.get_state_value_estimate(next_state_key)
                            
                            if state_value > best_value:
                                best_value = state_value
                                best_action = a
                    except:
                        pass  # En cas d'erreur, garder l'action par défaut
                    
                    temp_env.close()
                
                action = best_action
                self.bias_usage.append(1)  # Biais utilisé
            
            obs, reward, terminated, truncated, _ = env_copy.step(action)
            total_reward += reward
            steps += 1
        
        env_copy.close()
        
        # Mettre à jour la mémoire avec le résultat
        self.update_state_memory(state_key, total_reward)
        
        return total_reward
    
    def adaptive_simulation_count(self, state, action):
        """
        Détermine le nombre de simulations adaptatif basé sur la confiance
        """
        state_key = self.get_state_key(state)
        confidence = self.get_state_confidence(state_key)
        
        # Plus la confiance est faible, plus on fait de simulations
        if confidence < self.confidence_threshold:
            adaptive_sims = int(self.n_simulations * 1.5)  # +50% de simulations
        elif confidence > 0.8:
            adaptive_sims = int(self.n_simulations * 0.7)  # -30% de simulations
        else:
            adaptive_sims = self.n_simulations
        
        self.simulation_counts.append(adaptive_sims)
        self.confidence_scores.append(confidence)
        
        return adaptive_sims
    
    def evaluate_action_optimized(self, state, action):
        """
        Évalue une action avec les améliorations optimisées
        """
        state_key = self.get_state_key(state)
        
        # Nombre adaptatif de simulations
        n_sims = self.adaptive_simulation_count(state, action)
        
        rewards = []
        for _ in range(n_sims):
            # Utiliser le playout biaisé
            reward = self.biased_playout(state, action)
            rewards.append(reward)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # Appliquer le biais de valeur d'état si disponible
        state_value_bias = self.get_state_value_estimate(state_key)
        if state_value_bias != 0.0:
            biased_reward = (1 - self.value_bias_weight) * mean_reward + \
                           self.value_bias_weight * state_value_bias
        else:
            biased_reward = mean_reward
        
        return biased_reward, std_reward, n_sims
    
    def select_action(self, state):
        """
        Sélectionne la meilleure action avec les optimisations
        """
        action_values = {}
        action_stds = {}
        action_sim_counts = {}
        
        for action in range(self.action_space_size):
            mean_reward, std_reward, n_sims = self.evaluate_action_optimized(state, action)
            action_values[action] = mean_reward
            action_stds[action] = std_reward
            action_sim_counts[action] = n_sims
        
        # Sélectionner la meilleure action
        best_action = max(action_values, key=action_values.get)
        
        return best_action, action_values, action_stds, action_sim_counts

class OptimizedMCTSEvaluator:
    """
    Évaluateur pour comparer MCTS standard vs MCTS optimisé
    """
    
    def __init__(self):
        self.results = {}
    
    def compare_algorithms(self, env_name: str = "FrozenLake-v1", 
                          n_episodes: int = 100, n_simulations: int = 500,
                          env_kwargs: Optional[Dict] = None, result_label: Optional[str] = None):
        """
        Compare Flat MCTS vs MCTS Optimisé
        """
        print(f"\n=== Comparaison MCTS sur {env_name} ===")
        
        from phase1_claude import FlatMCTS  # Assurez-vous que le fichier est accessible
        
        if env_kwargs is None:
            env_kwargs = {}
        env = gym.make(env_name, **env_kwargs)
        
        # Initialiser les algorithmes
        flat_mcts = FlatMCTS(env, n_simulations=n_simulations)
        optimized_mcts = OptimizedMCTS(env, n_simulations=n_simulations)
        
        # Résultats
        flat_results = {'scores': [], 'times': [], 'sim_counts': []}
        opt_results = {'scores': [], 'times': [], 'sim_counts': [], 
                      'confidence_scores': [], 'bias_usage': []}
        
        print("Test Flat MCTS...")
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_time = 0
            
            while True:
                if hasattr(env.unwrapped, 's'):
                    current_state = env.unwrapped.s
                elif hasattr(env.unwrapped, 'state'):
                    current_state = env.unwrapped.state
                else:
                    current_state = state
                
                start_time = time.time()
                action, _, _ = flat_mcts.select_action(current_state)
                episode_time += time.time() - start_time
                
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            flat_results['scores'].append(total_reward)
            flat_results['times'].append(episode_time)
            flat_results['sim_counts'].append(n_simulations * flat_mcts.action_space_size)
            
            if (episode + 1) % 20 == 0:
                print(f"  Épisode {episode + 1}/{n_episodes} - Score moyen: {np.mean(flat_results['scores'][-20:]):.2f}")
        
        print("Test MCTS Optimisé...")
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_time = 0
            episode_sim_count = 0
            
            while True:
                if hasattr(env.unwrapped, 's'):
                    current_state = env.unwrapped.s
                elif hasattr(env.unwrapped, 'state'):
                    current_state = env.unwrapped.state
                else:
                    current_state = state
                
                start_time = time.time()
                action, _, _, sim_counts = optimized_mcts.select_action(current_state)
                episode_time += time.time() - start_time
                
                episode_sim_count += sum(sim_counts.values())
                
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            opt_results['scores'].append(total_reward)
            opt_results['times'].append(episode_time)
            opt_results['sim_counts'].append(episode_sim_count)
            
            if (episode + 1) % 20 == 0:
                print(f"  Épisode {episode + 1}/{n_episodes} - Score moyen: {np.mean(opt_results['scores'][-20:]):.2f}")
        
        # Ajouter les statistiques d'optimisation
        opt_results['confidence_scores'] = optimized_mcts.confidence_scores
        opt_results['bias_usage'] = optimized_mcts.bias_usage
        
        env.close()
        
        # Calculer les statistiques de comparaison
        comparison_results = {
            'env_name': env_name,
            'flat_mcts': {
                'avg_score': np.mean(flat_results['scores']),
                'std_score': np.std(flat_results['scores']),
                'success_rate': sum(1 for s in flat_results['scores'] if s > 0) / len(flat_results['scores']) * 100,
                'avg_time': np.mean(flat_results['times']),
                'avg_simulations': np.mean(flat_results['sim_counts']),
                'scores': flat_results['scores'],
                'times': flat_results['times']
            },
            'optimized_mcts': {
                'avg_score': np.mean(opt_results['scores']),
                'std_score': np.std(opt_results['scores']),
                'success_rate': sum(1 for s in opt_results['scores'] if s > 0) / len(opt_results['scores']) * 100,
                'avg_time': np.mean(opt_results['times']),
                'avg_simulations': np.mean(opt_results['sim_counts']),
                'avg_confidence': np.mean(opt_results['confidence_scores']) if opt_results['confidence_scores'] else 0,
                'bias_usage_rate': np.mean(opt_results['bias_usage']) * 100 if opt_results['bias_usage'] else 0,
                'scores': opt_results['scores'],
                'times': opt_results['times'],
                'confidence_scores': opt_results['confidence_scores'],
                'sim_counts': opt_results['sim_counts']
            }
        }
        
        if result_label is None:
            result_label = env_name
        self.results[result_label] = comparison_results
        
        # Afficher les résultats
        print(f"\n=== RÉSULTATS COMPARAISON {env_name} ===")
        print(f"Flat MCTS:")
        print(f"  Score moyen: {comparison_results['flat_mcts']['avg_score']:.3f} ± {comparison_results['flat_mcts']['std_score']:.3f}")
        print(f"  Taux de succès: {comparison_results['flat_mcts']['success_rate']:.1f}%")
        print(f"  Temps moyen: {comparison_results['flat_mcts']['avg_time']:.3f}s")
        print(f"  Simulations moyennes: {comparison_results['flat_mcts']['avg_simulations']:.0f}")
        
        print(f"\nMCTS Optimisé:")
        print(f"  Score moyen: {comparison_results['optimized_mcts']['avg_score']:.3f} ± {comparison_results['optimized_mcts']['std_score']:.3f}")
        print(f"  Taux de succès: {comparison_results['optimized_mcts']['success_rate']:.1f}%")
        print(f"  Temps moyen: {comparison_results['optimized_mcts']['avg_time']:.3f}s")
        print(f"  Simulations moyennes: {comparison_results['optimized_mcts']['avg_simulations']:.0f}")
        print(f"  Confiance moyenne: {comparison_results['optimized_mcts']['avg_confidence']:.3f}")
        print(f"  Utilisation du biais: {comparison_results['optimized_mcts']['bias_usage_rate']:.1f}%")
        
        # Calcul de l'amélioration
        score_improvement = ((comparison_results['optimized_mcts']['avg_score'] - 
                             comparison_results['flat_mcts']['avg_score']) / 
                            comparison_results['flat_mcts']['avg_score'] * 100)
        
        efficiency_improvement = ((comparison_results['flat_mcts']['avg_simulations'] - 
                                  comparison_results['optimized_mcts']['avg_simulations']) / 
                                 comparison_results['flat_mcts']['avg_simulations'] * 100)
        
        print(f"\nAméliorations:")
        print(f"  Score: {score_improvement:+.1f}%")
        print(f"  Efficacité (moins de simulations): {efficiency_improvement:+.1f}%")
        
        return comparison_results
    
    def plot_comparison(self, env_name: str):
        """
        Affiche les graphiques de comparaison
        """
        if env_name not in self.results:
            print(f"Pas de résultats pour {env_name}")
            return
        
        results = self.results[env_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comparaison MCTS - {env_name}', fontsize=16)
        
        # 1. Évolution des scores
        ax1 = axes[0, 0]
        episodes = range(1, len(results['flat_mcts']['scores']) + 1)
        
        ax1.plot(episodes, results['flat_mcts']['scores'], 'b-', alpha=0.6, label='Flat MCTS')
        ax1.plot(episodes, results['optimized_mcts']['scores'], 'r-', alpha=0.6, label='MCTS Optimisé')
        
        # Moyennes mobiles
        window = min(10, len(episodes) // 5)
        if window > 1:
            flat_ma = np.convolve(results['flat_mcts']['scores'], np.ones(window)/window, mode='valid')
            opt_ma = np.convolve(results['optimized_mcts']['scores'], np.ones(window)/window, mode='valid')
            
            ax1.plot(range(window, len(episodes) + 1), flat_ma, 'b--', linewidth=2, label='Flat MCTS (MA)')
            ax1.plot(range(window, len(episodes) + 1), opt_ma, 'r--', linewidth=2, label='Optimisé (MA)')
        
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Score')
        ax1.set_title('Évolution des scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution des scores
        ax2 = axes[0, 1]
        ax2.hist(results['flat_mcts']['scores'], bins=15, alpha=0.6, color='blue', 
                label=f'Flat MCTS (μ={results["flat_mcts"]["avg_score"]:.3f})', density=True)
        ax2.hist(results['optimized_mcts']['scores'], bins=15, alpha=0.6, color='red', 
                label=f'Optimisé (μ={results["optimized_mcts"]["avg_score"]:.3f})', density=True)
        
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Densité')
        ax2.set_title('Distribution des scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparaison des métriques principales
        ax3 = axes[0, 2]
        metrics = ['Score moyen', 'Taux succès (%)', 'Temps (s)']
        flat_values = [
            results['flat_mcts']['avg_score'],
            results['flat_mcts']['success_rate'],
            results['flat_mcts']['avg_time']
        ]
        opt_values = [
            results['optimized_mcts']['avg_score'],
            results['optimized_mcts']['success_rate'],
            results['optimized_mcts']['avg_time']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, flat_values, width, label='Flat MCTS', alpha=0.7, color='blue')
        bars2 = ax3.bar(x + width/2, opt_values, width, label='MCTS Optimisé', alpha=0.7, color='red')
        
        ax3.set_xlabel('Métriques')
        ax3.set_ylabel('Valeur')
        ax3.set_title('Comparaison des métriques')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Nombre de simulations par épisode
        ax4 = axes[1, 0]
        ax4.plot(episodes, [results['flat_mcts']['avg_simulations']] * len(episodes), 
                'b-', linewidth=2, label='Flat MCTS (constant)')
        ax4.plot(episodes, results['optimized_mcts']['sim_counts'], 
                'r-', alpha=0.7, label='MCTS Optimisé (adaptatif)')
        
        ax4.set_xlabel('Épisode')
        ax4.set_ylabel('Nombre de simulations')
        ax4.set_title('Simulations par épisode')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Évolution de la confiance (MCTS Optimisé seulement)
        ax5 = axes[1, 1]
        if results['optimized_mcts']['confidence_scores']:
            confidence_scores = results['optimized_mcts']['confidence_scores']
            ax5.plot(range(len(confidence_scores)), confidence_scores, 'g-', alpha=0.7)
            ax5.axhline(y=np.mean(confidence_scores), color='r', linestyle='--', 
                       label=f'Moyenne: {np.mean(confidence_scores):.3f}')
            ax5.set_xlabel('Décision')
            ax5.set_ylabel('Score de confiance')
            ax5.set_title('Évolution de la confiance')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Pas de données\nde confiance', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Confiance non disponible')
        
        # 6. Efficacité comparative
        ax6 = axes[1, 2]
        
        # Calculer l'efficacité (score / simulations)
        flat_efficiency = results['flat_mcts']['avg_score'] / results['flat_mcts']['avg_simulations']
        opt_efficiency = results['optimized_mcts']['avg_score'] / results['optimized_mcts']['avg_simulations']
        
        algorithms = ['Flat MCTS', 'MCTS Optimisé']
        efficiencies = [flat_efficiency, opt_efficiency]
        
        bars = ax6.bar(algorithms, efficiencies, color=['blue', 'red'], alpha=0.7)
        ax6.set_ylabel('Efficacité (Score/Simulations)')
        ax6.set_title('Efficacité comparative')
        
        for bar, eff in zip(bars, efficiencies):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{eff:.6f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'phase3_mcts_comparison_{env_name}.png', dpi=300)
        # plt.show()
    
    def analyze_optimization_impact(self, env_name: str):
        """
        Analyse détaillée de l'impact des optimisations
        """
        if env_name not in self.results:
            print(f"Pas de résultats pour {env_name}")
            return
        
        results = self.results[env_name]
        
        print(f"\n=== ANALYSE DÉTAILLÉE DES OPTIMISATIONS - {env_name} ===")
        
        # 1. Amélioration du score
        score_improvement = ((results['optimized_mcts']['avg_score'] - 
                             results['flat_mcts']['avg_score']) / 
                            abs(results['flat_mcts']['avg_score']) * 100)
        
        print(f"1. Amélioration du score: {score_improvement:+.2f}%")
        
        # 2. Efficacité computationnelle
        sim_reduction = ((results['flat_mcts']['avg_simulations'] - 
                         results['optimized_mcts']['avg_simulations']) / 
                        results['flat_mcts']['avg_simulations'] * 100)
        
        print(f"2. Réduction des simulations: {sim_reduction:+.2f}%")
        
        # 3. Temps d'exécution
        time_ratio = results['optimized_mcts']['avg_time'] / results['flat_mcts']['avg_time']
        print(f"3. Ratio de temps d'exécution: {time_ratio:.2f}x")
        
        # 4. Consistance (variance des scores)
        flat_cv = results['flat_mcts']['std_score'] / abs(results['flat_mcts']['avg_score'])
        opt_cv = results['optimized_mcts']['std_score'] / abs(results['optimized_mcts']['avg_score'])
        consistency_improvement = (flat_cv - opt_cv) / flat_cv * 100
        
        print(f"4. Amélioration de la consistance: {consistency_improvement:+.2f}%")
        
        # 5. Utilisation des optimisations
        if results['optimized_mcts']['bias_usage_rate'] > 0:
            print(f"5. Utilisation du biais: {results['optimized_mcts']['bias_usage_rate']:.1f}% des décisions")
        
        if results['optimized_mcts']['avg_confidence'] > 0:
            print(f"6. Confiance moyenne: {results['optimized_mcts']['avg_confidence']:.3f}")
        
        # 7. Test statistique simple (différence significative)
        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_ind(results['flat_mcts']['scores'], 
                                            results['optimized_mcts']['scores'])
            print(f"7. Test t (p-value): {p_value:.6f} {'(significatif)' if p_value < 0.05 else '(non significatif)'}")
        except ImportError:
            print("7. Scipy non disponible pour test statistique")
        
        return {
            'score_improvement': score_improvement,
            'simulation_reduction': sim_reduction,
            'time_ratio': time_ratio,
            'consistency_improvement': consistency_improvement
        }

# Version simplifiée de FlatMCTS pour éviter les dépendances
class SimpleFlatMCTS:
    """Version simplifiée du Flat MCTS pour la comparaison"""
    
    def __init__(self, env, n_simulations: int = 1000, max_depth: int = 100):
        self.env = env
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.action_space_size = env.action_space.n
    
    def random_playout(self, state, action):
        env_copy = gym.make(self.env.spec.id)
        env_copy.reset()
        
        if hasattr(env_copy.unwrapped, 's'):
            env_copy.unwrapped.s = state
        elif hasattr(env_copy.unwrapped, 'state'):
            env_copy.unwrapped.state = state
        
        obs, reward, terminated, truncated, _ = env_copy.step(action)
        total_reward = reward
        
        steps = 0
        while not (terminated or truncated) and steps < self.max_depth:
            random_action = env_copy.action_space.sample()
            obs, reward, terminated, truncated, _ = env_copy.step(random_action)
            total_reward += reward
            steps += 1
        
        env_copy.close()
        return total_reward
    
    def select_action(self, state):
        action_values = {}
        
        for action in range(self.action_space_size):
            rewards = [self.random_playout(state, action) for _ in range(self.n_simulations)]
            action_values[action] = np.mean(rewards)
        
        best_action = max(action_values, key=action_values.get)
        return best_action, action_values, {}

def main():
    """
    Fonction principale pour tester le MCTS optimisé
    """
    print("=== Phase 3: MCTS Optimisé (Recherche) ===")
    print("Comparaison Flat MCTS vs MCTS Optimisé basé sur l'article de recherche\n")

    evaluator = OptimizedMCTSEvaluator()
    
    # Test sur FrozenLake avec is_slippery=True
    print("Test 1: FrozenLake-v1 (slippery=True)")
    results1 = evaluator.compare_algorithms(
        env_name="FrozenLake-v1",
        n_episodes=100,
        n_simulations=300,
        env_kwargs={'is_slippery': True},
        result_label="FrozenLake-v1 (slippery=True)"
    )
    evaluator.plot_comparison("FrozenLake-v1 (slippery=True)")
    evaluator.analyze_optimization_impact("FrozenLake-v1 (slippery=True)")
    
    # Test sur FrozenLake avec is_slippery=False
    print("\n" + "="*50)
    print("Test 2: FrozenLake-v1 (slippery=False)")
    results2 = evaluator.compare_algorithms(
        env_name="FrozenLake-v1",
        n_episodes=100,
        n_simulations=300,
        env_kwargs={'is_slippery': False},
        result_label="FrozenLake-v1 (slippery=False)"
    )
    evaluator.plot_comparison("FrozenLake-v1 (slippery=False)")
    evaluator.analyze_optimization_impact("FrozenLake-v1 (slippery=False)")
    
    # ...vous pouvez continuer avec d'autres tests ou résumés...

if __name__ == "__main__":
    main()