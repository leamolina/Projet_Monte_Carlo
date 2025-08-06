import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
import math
import random

class MCTSNode:
    """
    Nœud de l'arbre MCTS
    """
    def __init__(self, state, parent=None, action=None, untried_actions=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action qui a mené à ce nœud
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value = 0.0
        self.untried_actions = untried_actions if untried_actions is not None else []
        
    def is_fully_expanded(self):
        """Vérifie si tous les enfants ont été créés"""
        # Correction robuste pour éviter l'erreur si untried_actions est None
        return not self.untried_actions
    
    def best_child(self, c_param=1.4):
        """Sélectionne le meilleur enfant selon UCB1"""
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_actions):
        """Politique de rollout (aléatoire par défaut)"""
        return random.choice(possible_actions)
    
    def select_action(self):
        """Sélectionne l'action avec le plus de visites"""
        if not self.children:
            return None
        
        visits = [(action, child.visits) for action, child in self.children.items()]
        return max(visits, key=lambda x: x[1])[0]

class UCTMCTSAgent:
    """
    Agent MCTS complet avec UCT (Upper Confidence Bounds applied to Trees)
    """
    
    def __init__(self, env, n_simulations: int = 1000, c_param: float = 1.4, 
                 max_depth: int = 100, tree_reuse: bool = False):
        """
        Args:
            env: Environnement Gymnasium
            n_simulations: Nombre de simulations MCTS
            c_param: Paramètre d'exploration UCB (√2 théorique)
            max_depth: Profondeur maximale des rollouts
            tree_reuse: Réutiliser l'arbre entre les actions
        """
        self.env = env
        self.n_simulations = n_simulations
        self.c_param = c_param
        self.max_depth = max_depth
        self.tree_reuse = tree_reuse
        self.action_space_size = env.action_space.n
        
        # Statistiques
        self.tree_sizes = []
        self.tree_depths = []
        self.selection_depths = []
        self.rollout_lengths = []
        
        # Arbre racine pour réutilisation
        self.root = None
    
    def get_state_key(self, state):
        """Convertit un état en clé hashable"""
        if isinstance(state, (int, float)):
            return state
        elif isinstance(state, (list, np.ndarray)):
            return tuple(state)
        elif hasattr(self.env.unwrapped, 's'):  # FrozenLake
            return self.env.unwrapped.s
        else:
            return str(state)
    
    def get_possible_actions(self, state):
        """Retourne les actions possibles depuis un état"""
        return list(range(self.action_space_size))
    
    def simulate_action(self, state, action):
        """
        Simule une action depuis un état donné
        Retourne: (next_state, reward, done)
        """
        # Créer une copie de l'environnement
        env_copy = gym.make(self.env.spec.id)
        env_copy.reset()
        
        # Restaurer l'état
        if hasattr(env_copy.unwrapped, 's'):  # FrozenLake
            env_copy.unwrapped.s = state
        elif hasattr(env_copy.unwrapped, 'state'):  # CartPole, LunarLander
            env_copy.unwrapped.state = state
        
        # Exécuter l'action
        next_obs, reward, terminated, truncated, _ = env_copy.step(action)
        
        # Obtenir le nouvel état
        if hasattr(env_copy.unwrapped, 's'):
            next_state = env_copy.unwrapped.s
        elif hasattr(env_copy.unwrapped, 'state'):
            next_state = env_copy.unwrapped.state
        else:
            next_state = next_obs
        
        env_copy.close()
        return next_state, reward, terminated or truncated
    
    def rollout(self, state, depth=0):
        """
        Phase de simulation: rollout biaisé ε-greedy depuis un état
        """
        if depth >= self.max_depth:
            return 0.0
        
        env_copy = gym.make(self.env.spec.id)
        env_copy.reset()
        if hasattr(env_copy.unwrapped, 's'):
            env_copy.unwrapped.s = state
        elif hasattr(env_copy.unwrapped, 'state'):
            env_copy.unwrapped.state = state
        
        total_reward = 0.0
        rollout_depth = 0
        epsilon = 0.2  # Biais : 20% aléatoire, 80% heuristique
        
        while rollout_depth < self.max_depth - depth:
            if np.random.rand() < epsilon:
                action = env_copy.action_space.sample()
            else:
                # Heuristique simple : aller à droite (2) ou en bas (1) si possible
                action = np.random.choice([1, 2])
            obs, reward, terminated, truncated, _ = env_copy.step(action)
            total_reward += reward
            rollout_depth += 1
            if terminated or truncated:
                break
        
        env_copy.close()
        self.rollout_lengths.append(rollout_depth)
        return total_reward
    
    def select(self, node):
        """
        Phase de sélection: descendre dans l'arbre selon UCB1
        """
        path = [node]
        depth = 0
        
        while not node.is_fully_expanded() or node.children:
            if not node.is_fully_expanded():
                # Il reste des actions à essayer
                break
            else:
                # Sélectionner le meilleur enfant selon UCB1
                node = node.best_child(self.c_param)
                path.append(node)
                depth += 1
                
                # Vérifier si on a atteint un état terminal
                if depth > self.max_depth:
                    break
        
        self.selection_depths.append(depth)
        return node, path
    
    def expand(self, node):
        """
        Phase d'expansion: ajouter un nouveau nœud enfant
        """
        if node.untried_actions is None:
            # Première expansion de ce nœud
            node.untried_actions = self.get_possible_actions(node.state)
        
        if not node.untried_actions:
            return node  # Aucune action à essayer
        
        # Choisir une action non essayée
        action = node.untried_actions.pop()
        
        # Simuler l'action pour obtenir le nouvel état
        next_state, reward, done = self.simulate_action(node.state, action)
        
        # Créer le nouveau nœud enfant
        child_node = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child_node
        
        return child_node
    
    def backpropagate(self, path, reward):
        """
        Phase de rétropropagation: mettre à jour les statistiques
        """
        for node in path:
            node.visits += 1
            node.value += reward
    
    def search(self, root_state):
        """
        Recherche MCTS complète depuis un état racine
        """
        # Créer ou réutiliser le nœud racine
        if not self.tree_reuse or self.root is None:
            self.root = MCTSNode(root_state)
        
        root = self.root
        
        # Effectuer les simulations MCTS
        for simulation in range(self.n_simulations):
            # 1. Sélection
            node, path = self.select(root)
            
            # 2. Expansion
            if not node.is_fully_expanded() and node.visits > 0:
                node = self.expand(node)
                path.append(node)
            
            # 3. Simulation (Rollout)
            reward = self.rollout(node.state, len(path))
            
            # 4. Rétropropagation
            self.backpropagate(path, reward)
        
        # Collecter les statistiques de l'arbre
        self._collect_tree_stats(root)
        
        return root
    
    def _collect_tree_stats(self, root):
        """Collecte les statistiques de l'arbre"""
        def count_nodes(node, depth=0):
            count = 1  # Le nœud actuel
            max_depth = depth
            
            for child in node.children.values():
                child_count, child_depth = count_nodes(child, depth + 1)
                count += child_count
                max_depth = max(max_depth, child_depth)
            
            return count, max_depth
        
        tree_size, tree_depth = count_nodes(root)
        self.tree_sizes.append(tree_size)
        self.tree_depths.append(tree_depth)
    
    def select_action(self, state):
        """
        Sélectionne la meilleure action en utilisant MCTS avec UCT
        """
        root = self.search(state)
        
        if not root.children:
            # Aucun enfant trouvé, action aléatoire
            return random.randint(0, self.action_space_size - 1), {}, {}
        
        # Sélectionner l'action avec le plus de visites
        best_action = root.select_action()
        
        # Collecter les statistiques des actions
        action_values = {}
        action_visits = {}
        
        for action, child in root.children.items():
            action_values[action] = child.value / child.visits if child.visits > 0 else 0
            action_visits[action] = child.visits
        
        # Mise à jour de l'arbre pour réutilisation
        if self.tree_reuse and best_action in root.children:
            self.root = root.children[best_action]
            self.root.parent = None
        
        return best_action, action_values, action_visits

class UCTMCTSEvaluator:
    """
    Évaluateur pour le MCTS avec UCT
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_uct_mcts(self, env_name: str = "FrozenLake-v1", 
                         n_episodes: int = 50, n_simulations: int = 500,
                         c_param: float = 1.4,
                         env_kwargs: Optional[Dict] = None,
                         result_label: Optional[str] = None):
        """
        Évalue les performances d'UCT MCTS
        """
        print(f"\n=== Évaluation UCT MCTS sur {env_name} ===")
        print(f"Épisodes: {n_episodes}, Simulations: {n_simulations}, C: {c_param}")
        
        if env_kwargs is None:
            env_kwargs = {}
        env = gym.make(env_name, **env_kwargs)
        agent = UCTMCTSAgent(env, n_simulations=n_simulations, c_param=c_param)
        
        scores = []
        episode_lengths = []
        computation_times = []
        tree_stats = {'sizes': [], 'depths': [], 'selection_depths': [], 'rollout_lengths': []}
        
        success_count = 0
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_start = time.time()
            
            while True:
                # Obtenir l'état actuel
                if hasattr(env.unwrapped, 's'):
                    current_state = env.unwrapped.s
                elif hasattr(env.unwrapped, 'state'):
                    current_state = env.unwrapped.state
                else:
                    current_state = state
                
                # Sélection d'action avec mesure de temps
                action_start = time.time()
                action, action_values, action_visits = agent.select_action(current_state)
                action_time = time.time() - action_start
                computation_times.append(action_time)
                
                # Exécuter l'action
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            scores.append(total_reward)
            episode_lengths.append(steps)
            
            # Critère de succès
            if env_name == "FrozenLake-v1":
                success_count += (total_reward > 0)
            elif env_name == "CartPole-v1":
                success_count += (total_reward >= 195)
            elif env_name == "LunarLander-v2":
                success_count += (total_reward >= 200)
            
            if (episode + 1) % 10 == 0:
                avg_score = np.mean(scores[-10:])
                print(f"Épisode {episode + 1}/{n_episodes} - Score moyen (10 derniers): {avg_score:.2f}")
        
        # Collecter les statistiques d'arbre
        tree_stats['sizes'] = agent.tree_sizes
        tree_stats['depths'] = agent.tree_depths
        tree_stats['selection_depths'] = agent.selection_depths
        tree_stats['rollout_lengths'] = agent.rollout_lengths
        
        env.close()
        
        # Calculer les statistiques finales
        results = {
            'env_name': env_name,
            'n_simulations': n_simulations,
            'c_param': c_param,
            'scores': scores,
            'episode_lengths': episode_lengths,
            'computation_times': computation_times,
            'tree_stats': tree_stats,
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'success_rate': success_count / n_episodes * 100,
            'avg_episode_length': np.mean(episode_lengths),
            'avg_computation_time': np.mean(computation_times),
            'avg_tree_size': np.mean(tree_stats['sizes']) if tree_stats['sizes'] else 0,
            'avg_tree_depth': np.mean(tree_stats['depths']) if tree_stats['depths'] else 0,
            'avg_selection_depth': np.mean(tree_stats['selection_depths']) if tree_stats['selection_depths'] else 0,
            'avg_rollout_length': np.mean(tree_stats['rollout_lengths']) if tree_stats['rollout_lengths'] else 0
        };
        
        if result_label is None:
            result_label = env_name
        self.results[result_label] = results;
        
        print(f"\nRésultats UCT MCTS ({env_name}):")
        print(f"  Score moyen: {results['avg_score']:.3f} ± {results['std_score']:.3f}")
        print(f"  Taux de succès: {results['success_rate']:.1f}%")
        print(f"  Temps de calcul moyen: {results['avg_computation_time']:.3f}s")
        print(f"  Taille d'arbre moyenne: {results['avg_tree_size']:.1f} nœuds")
        print(f"  Profondeur d'arbre moyenne: {results['avg_tree_depth']:.1f}")
        print(f"  Profondeur de sélection moyenne: {results['avg_selection_depth']:.1f}")
        print(f"  Longueur de rollout moyenne: {results['avg_rollout_length']:.1f}")
        
        return results
    
    def compare_c_parameters(self, env_name: str = "FrozenLake-v1", 
                           c_values: List[float] = [0.5, 1.0, 1.4, 2.0],
                           n_episodes: int = 30, n_simulations: int = 300):
        """
        Compare différentes valeurs du paramètre C d'UCB1
        """
        print(f"\n=== Comparaison paramètres C pour UCT MCTS ({env_name}) ===")
        
        comparison_results = {}
        
        for c in c_values:
            print(f"\nTest avec C = {c}")
            results = self.evaluate_uct_mcts(
                env_name=env_name,
                n_episodes=n_episodes,
                n_simulations=n_simulations,
                c_param=c
            )
            comparison_results[c] = results
        
        self._plot_c_comparison(comparison_results, env_name)
        return comparison_results
    
    def _plot_c_comparison(self, comparison_results: Dict, env_name: str):
        """
        Affiche la comparaison des paramètres C
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comparaison des paramètres C - UCT MCTS ({env_name})', fontsize=16)
        
        c_values = list(comparison_results.keys())
        
        # 1. Score moyen
        ax1 = axes[0, 0]
        scores = [comparison_results[c]['avg_score'] for c in c_values]
        error_bars = [comparison_results[c]['std_score'] for c in c_values]
        
        bars = ax1.bar(range(len(c_values)), scores, yerr=error_bars, 
                      alpha=0.7, capsize=5, color='skyblue')
        ax1.set_xlabel('Paramètre C')
        ax1.set_ylabel('Score moyen')
        ax1.set_title('Score moyen par paramètre C')
        ax1.set_xticks(range(len(c_values)))
        ax1.set_xticklabels([f'C={c}' for c in c_values])
        
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. Taux de succès
        ax2 = axes[0, 1]
        success_rates = [comparison_results[c]['success_rate'] for c in c_values]
        bars = ax2.bar(range(len(c_values)), success_rates, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Paramètre C')
        ax2.set_ylabel('Taux de succès (%)')
        ax2.set_title('Taux de succès par paramètre C')
        ax2.set_xticks(range(len(c_values)))
        ax2.set_xticklabels([f'C={c}' for c in c_values])
        
        for bar, rate in zip(bars, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Taille d'arbre moyenne
        ax3 = axes[0, 2]
        tree_sizes = [comparison_results[c]['avg_tree_size'] for c in c_values]
        bars = ax3.bar(range(len(c_values)), tree_sizes, alpha=0.7, color='orange')
        ax3.set_xlabel('Paramètre C')
        ax3.set_ylabel('Taille d\'arbre moyenne')
        ax3.set_title('Exploration de l\'arbre (taille)')
        ax3.set_xticks(range(len(c_values)))
        ax3.set_xticklabels([f'C={c}' for c in c_values])
        
        for bar, size in zip(bars, tree_sizes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{size:.0f}', ha='center', va='bottom')
        
        # 4. Profondeur d'arbre
        ax4 = axes[1, 0]
        tree_depths = [comparison_results[c]['avg_tree_depth'] for c in c_values]
        bars = ax4.bar(range(len(c_values)), tree_depths, alpha=0.7, color='coral')
        ax4.set_xlabel('Paramètre C')
        ax4.set_ylabel('Profondeur moyenne')
        ax4.set_title('Exploration de l\'arbre (profondeur)')
        ax4.set_xticks(range(len(c_values)))
        ax4.set_xticklabels([f'C={c}' for c in c_values])
        
        for bar, depth in zip(bars, tree_depths):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{depth:.1f}', ha='center', va='bottom')
        
        # 5. Temps de calcul
        ax5 = axes[1, 1]
        comp_times = [comparison_results[c]['avg_computation_time'] for c in c_values]
        bars = ax5.bar(range(len(c_values)), comp_times, alpha=0.7, color='mediumpurple')
        ax5.set_xlabel('Paramètre C')
        ax5.set_ylabel('Temps de calcul (s)')
        ax5.set_title('Temps de calcul moyen')
        ax5.set_xticks(range(len(c_values)))
        ax5.set_xticklabels([f'C={c}' for c in c_values])
        
        for bar, time_val in zip(bars, comp_times):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # 6. Distribution des scores pour chaque C
        ax6 = axes[1, 2]
        for i, c in enumerate(c_values):
            scores_dist = comparison_results[c]['scores']
            ax6.hist(scores_dist, bins=10, alpha=0.6, 
                    label=f'C={c} (μ={np.mean(scores_dist):.2f})', density=True)
        
        ax6.set_xlabel('Score')
        ax6.set_ylabel('Densité')
        ax6.set_title('Distribution des scores')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'phase4_claude_{env_name}.png', dpi=300)
    
    def plot_detailed_analysis(self, env_name: str, c_param: float = 1.4):
        """
        Analyse détaillée pour une configuration donnée
        """
        key = f"{env_name}_c{c_param}"
        if key not in self.results:
            print(f"Pas de résultats pour {env_name} avec C={c_param}")
            return
        
        results = self.results[key]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Analyse détaillée UCT MCTS - {env_name} (C={c_param})', fontsize=16)
        
        # 1. Évolution des scores
        ax1 = axes[0, 0]
        episodes = range(1, len(results['scores']) + 1)
        ax1.plot(episodes, results['scores'], 'b-', alpha=0.7, linewidth=1)
        
        # Moyenne mobile
        window = min(10, len(results['scores']) // 5)
        if window > 1:
            moving_avg = np.convolve(results['scores'], np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(results['scores']) + 1), moving_avg, 
                    'r-', linewidth=2, label=f'Moyenne mobile ({window})')
        
        ax1.axhline(y=results['avg_score'], color='g', linestyle='--', 
                   label=f'Moyenne: {results["avg_score"]:.3f}')
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Score')
        ax1.set_title('Évolution des scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Statistiques d'arbre par épisode
        ax2 = axes[0, 1]
        if results['tree_stats']['sizes']:
            ax2.plot(range(len(results['tree_stats']['sizes'])), 
                    results['tree_stats']['sizes'], 'g-', alpha=0.7, label='Taille')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(range(len(results['tree_stats']['depths'])), 
                         results['tree_stats']['depths'], 'orange', alpha=0.7, label='Profondeur')
            
            ax2.set_xlabel('Action')
            ax2.set_ylabel('Taille d\'arbre', color='g')
            ax2_twin.set_ylabel('Profondeur d\'arbre', color='orange')
            ax2.set_title('Évolution de l\'arbre MCTS')
            ax2.grid(True, alpha=0.3)
        
        # 3. Distribution des longueurs de rollout
        ax3 = axes[0, 2]
        if results['tree_stats']['rollout_lengths']:
            ax3.hist(results['tree_stats']['rollout_lengths'], bins=20, 
                    alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=results['avg_rollout_length'], color='red', linestyle='--',
                       label=f'Moyenne: {results["avg_rollout_length"]:.1f}')
            ax3.set_xlabel('Longueur de rollout')
            ax3.set_ylabel('Fréquence')
            ax3.set_title('Distribution des longueurs de rollout')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Temps de calcul par action
        ax4 = axes[1, 0]
        ax4.plot(range(len(results['computation_times'])), 
                results['computation_times'], 'purple', alpha=0.7)
        ax4.axhline(y=results['avg_computation_time'], color='red', linestyle='--',
                   label=f'Moyenne: {results["avg_computation_time"]:.3f}s')
        ax4.set_xlabel('Action')
        ax4.set_ylabel('Temps de calcul (s)')
        ax4.set_title('Temps de calcul par action')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Profondeur de sélection vs longueur de rollout
        ax5 = axes[1, 1]
        if results['tree_stats']['selection_depths'] and results['tree_stats']['rollout_lengths']:
            # Assurer que les deux listes ont la même longueur
            min_len = min(len(results['tree_stats']['selection_depths']), 
                         len(results['tree_stats']['rollout_lengths']))
            if min_len > 0:
                sel_depths = results['tree_stats']['selection_depths'][:min_len]
                roll_lengths = results['tree_stats']['rollout_lengths'][:min_len]
                
                ax5.scatter(sel_depths, roll_lengths, alpha=0.6, s=20)
                ax5.set_xlabel('Profondeur de sélection')
                ax5.set_ylabel('Longueur de rollout')
                ax5.set_title('Sélection vs Rollout')
                ax5.grid(True, alpha=0.3)
        
        # 6. Résumé des métriques
        ax6 = axes[1, 2]
        metrics = ['Score', 'Succès (%)', 'Temps (s)', 'Arbre (nœuds)', 'Profondeur']
        values = [
            results['avg_score'],
            results['success_rate'],
            results['avg_computation_time'] * 1000,  # en ms pour visibilité
            results['avg_tree_size'],
            results['avg_tree_depth']
        ]
        
        # Normaliser pour la visualisation
        normalized_values = []
        for i, val in enumerate(values):
            if i == 0:  # Score
                normalized_values.append(val * 100)  # Multiplier pour visibilité
            elif i == 1:  # Succès (déjà en %)
                normalized_values.append(val)
            elif i == 2:  # Temps (en ms)
                normalized_values.append(val)
            else:  # Arbre et profondeur
                normalized_values.append(val)
        
        bars = ax6.bar(metrics, normalized_values, alpha=0.7, 
                      color=['blue', 'green', 'red', 'orange', 'purple'])
        ax6.set_ylabel('Valeur (normalisée)')
        ax6.set_title('Résumé des métriques')
        plt.setp(ax6.get_xticklabels(), rotation=45)
        
        # Ajouter les vraies valeurs sur les barres
        true_labels = [f'{results["avg_score"]:.3f}', f'{results["success_rate"]:.1f}%', 
                      f'{results["avg_computation_time"]:.3f}s', f'{results["avg_tree_size"]:.0f}', 
                      f'{results["avg_tree_depth"]:.1f}']
        
        for bar, label in zip(bars, true_labels):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(normalized_values)*0.01,
                    label, ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def compare_all_algorithms(self, env_name: str = "FrozenLake-v1", 
                              n_episodes: int = 50, n_simulations: int = 400):
        """
        Compare tous les algorithmes MCTS développés
        """
        print(f"\n=== COMPARAISON FINALE - Tous les algorithmes MCTS ({env_name}) ===")
        
        from phase1_claude import FlatMCTS
        from phase3_claude import OptimizedMCTS
        
        env = gym.make(env_name)
        
        # Initialiser les algorithmes
        algorithms = {
            'Flat MCTS': FlatMCTS(env, n_simulations=n_simulations),
            'MCTS Optimisé': OptimizedMCTS(env, n_simulations=n_simulations),
            'UCT MCTS': UCTMCTSAgent(env, n_simulations=n_simulations, c_param=1.4)
        }
        
        final_results = {}
        
        for name, algorithm in algorithms.items():
            print(f"\nTest {name}...")
            
            scores = []
            times = []
            success_count = 0
            
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
                    
                    if name == 'UCT MCTS':
                        action, _, _ = algorithm.select_action(current_state)
                    else:
                        if name == 'MCTS Optimisé':
                            action, _, _, _ = algorithm.select_action(current_state)
                        else:  # Flat MCTS
                            action, _, _ = algorithm.select_action(current_state)
                    
                    episode_time += time.time() - start_time
                    
                    state, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                
                scores.append(total_reward)
                times.append(episode_time)
                
                if env_name == "FrozenLake-v1":
                    success_count += (total_reward > 0)
                elif env_name == "CartPole-v1":
                    success_count += (total_reward >= 195)
            
            final_results[name] = {
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'success_rate': success_count / n_episodes * 100,
                'avg_time': np.mean(times),
                'scores': scores
            }
        
        env.close()
        
        # Afficher la comparaison finale
        self._plot_final_comparison(final_results, env_name)
        
        return final_results
    
    def _plot_final_comparison(self, results: Dict, env_name: str):
        """
        Graphique de comparaison finale
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Comparaison finale des algorithmes MCTS - {env_name}', fontsize=16)
        
        algorithms = list(results.keys())
        
        # 1. Score moyen avec barres d'erreur
        ax1 = axes[0, 0]
        scores = [results[alg]['avg_score'] for alg in algorithms]
        errors = [results[alg]['std_score'] for alg in algorithms]
        
        bars = ax1.bar(algorithms, scores, yerr=errors, capsize=5, alpha=0.7, 
                      color=['blue', 'red', 'green'])
        ax1.set_ylabel('Score moyen')
        ax1.set_title('Performance moyenne')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. Taux de succès
        ax2 = axes[0, 1]
        success_rates = [results[alg]['success_rate'] for alg in algorithms]
        bars = ax2.bar(algorithms, success_rates, alpha=0.7, color=['blue', 'red', 'green'])
        ax2.set_ylabel('Taux de succès (%)')
        ax2.set_title('Fiabilité')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        for bar, rate in zip(bars, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Distribution des scores
        ax3 = axes[1, 0]
        for alg in algorithms:
            ax3.hist(results[alg]['scores'], bins=15, alpha=0.6, 
                    label=f'{alg} (μ={results[alg]["avg_score"]:.3f})', density=True)
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Densité')
        ax3.set_title('Distribution des performances')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Temps de calcul
        ax4 = axes[1, 1]
        times = [results[alg]['avg_time'] for alg in algorithms]
        bars = ax4.bar(algorithms, times, alpha=0.7, color=['blue', 'red', 'green'])
        ax4.set_ylabel('Temps moyen (s)')
        ax4.set_title('Efficacité computationnelle')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        for bar, time_val in zip(bars, times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'comparison_{env_name}.png', bbox_inches='tight')
        
        # Tableau de résumé
        print(f"\n=== TABLEAU DE RÉSUMÉ - {env_name} ===")
        print(f"{'Algorithme':<15} {'Score':<12} {'Succès (%)':<12} {'Temps (s)':<12}")
        print("-" * 55)
        for alg in algorithms:
            print(f"{alg:<15} {results[alg]['avg_score']:<12.3f} "
                  f"{results[alg]['success_rate']:<12.1f} {results[alg]['avg_time']:<12.3f}")

def main():
    """
    Fonction principale pour tester UCT MCTS
    """
    print("=== Phase 4: MCTS avec UCT (Arbre complet) ===")
    print("Implémentation complète de MCTS avec Upper Confidence Bounds applied to Trees\n")
    
    evaluator = UCTMCTSEvaluator()
    
    # 1. Évaluation sur FrozenLake avec is_slippery=True
    print("1. Évaluation UCT MCTS sur FrozenLake-v1 (slippery=True)")
    base_results_slippery = evaluator.evaluate_uct_mcts(
        env_name="FrozenLake-v1",
        n_episodes=50,
        n_simulations=2000,  # Amélioration : plus de simulations
        c_param=1.4,
        env_kwargs={'is_slippery': True},
        result_label="FrozenLake-v1 (slippery=True)"
    )
    evaluator.plot_detailed_analysis("FrozenLake-v1 (slippery=True)", 1.4)
    
    # 2. Évaluation sur FrozenLake avec is_slippery=False
    print("\n2. Évaluation UCT MCTS sur FrozenLake-v1 (slippery=False)")
    base_results_nonslippery = evaluator.evaluate_uct_mcts(
        env_name="FrozenLake-v1",
        n_episodes=50,
        n_simulations=2000,
        c_param=1.4,
        env_kwargs={'is_slippery': False},
        result_label="FrozenLake-v1 (slippery=False)"
    )
    evaluator.plot_detailed_analysis("FrozenLake-v1 (slippery=False)", 1.4)
    
    # 3. Comparaison des paramètres C
    print("\n3. Comparaison des paramètres C")
    c_comparison = evaluator.compare_c_parameters(
        env_name="FrozenLake-v1",
        c_values=[0.5, 1.0, 1.4, 2.0],
        n_episodes=30,
        n_simulations=300
    )
    
    # 4. Test sur CartPole
    print("\n4. Test sur CartPole-v1")
    cartpole_results = evaluator.evaluate_uct_mcts(
        env_name="CartPole-v1",
        n_episodes=30,
        n_simulations=200,
        c_param=1.4
    )
    
    # 5. Comparaison finale de tous les algorithmes
    print("\n5. Comparaison finale de tous les algorithmes MCTS")
    try:
        final_comparison = evaluator.compare_all_algorithms(
            env_name="FrozenLake-v1",
            n_episodes=40,
            n_simulations=300
        )
    except ImportError as e:
        print(f"Impossible d'importer tous les algorithmes: {e}")
        print("Assurez-vous que les fichiers des phases précédentes sont disponibles")

if __name__ == "__main__":
    main()