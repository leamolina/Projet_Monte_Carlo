# 🎓 Projet MCTS & RL - Recherche Monte Carlo et Jeux

## 🎯 Objectif général

Comparer différentes approches de décision (MCTS et Q-learning) sur plusieurs environnements Gymnasium. Proposer une amélioration issue d’un article de recherche externe dans une phase finale.

---

## 🧩 Environnements utilisés

| Jeu                 | Type                     | Rendu "human" | Intérêt pédagogique                            |
|---------------------|--------------------------|---------------|-------------------------------------------------|
| **FrozenLake-v1**   | Discret, stochastique    | ✅             | Environnement simple, idéal pour debug + RL     |
| **CartPole-v1**     | Discret, dynamique       | ✅             | Apprentissage dynamique, scores continus        |
| **LunarLander-v2**  | Discret, plus complexe   | ✅             | Défis stratégiques et contrôle spatial          |

---

## 🔶 Phase 1 — Flat Monte Carlo Tree Search

### ✅ Objectif
Utiliser **Flat MCTS**, c’est-à-dire une recherche par playouts depuis chaque action de la racine.

### 📌 Tâches
- Implémentation du Flat MCTS
- Évaluation de chaque action via plusieurs playouts aléatoires
- Exécution sur 3 jeux Gymnasium

### 🔬 Métriques
- Score moyen par épisode
- Taux de réussite (ex : FrozenLake)
- Temps d’exécution
- Nombre de simulations

---

## 🔶 Phase 2 — Q-Learning (tabulaire)

### ✅ Objectif
Implémenter une approche **Reinforcement Learning** avec Q-table pour comparer à MCTS.

### 📌 Tâches
- Implémentation du **Q-Learning** tabulaire
- Politique ε-greedy
- Exploration de paramètres (taux d’apprentissage, ε, discount)

### 📍 Environnements :
- Principalement `FrozenLake-v1`
- Optionnel : CartPole avec discretisation des états

### 🔬 Évaluation
- Convergence de la Q-table
- Politique apprise vs MCTS
- Score final moyen
- Nombre d’épisodes pour atteindre la stabilité

---

## 🔶 Phase 3 — MCTS optimisé (à partir d’un article de recherche)

### ✅ Objectif
Améliorer l’algorithme MCTS de la Phase 1 en se basant sur un **papier de recherche externe** non traité en cours. : Optimized Monte Carlo Tree Search for Enhanced Decision Making in the FrozenLake Environment
Esteban Aldana
Computer Science Department, Universidad del Valle de Guatemala Email: ald20591@uvg.edu.gt



### 📌 Exemples d’améliorations possibles :
- Playout biaisé : ε-greedy, heuristique, ou basée sur valeur estimée `V(s)`
- MCTS avec politique adaptative ou mémoire statistique
- Pruning ou réutilisation partielle d’arbre

### 🔬 Comparaison :
- Flat MCTS vs MCTS amélioré
- Évaluation sur `FrozenLake-v1`
- Impact sur taux de réussite et nombre de simulations nécessaires

---

## 🔶 Phase 4 — MCTS avec arbre (UCT)

### ✅ Objectif
Implémenter le **MCTS complet avec UCT** (Upper Confidence Bounds applied to Trees).

### 📌 Tâches
- Implémentation des 4 phases classiques :
  - Selection (UCB)
  - Expansion
  - Simulation
  - Backpropagation
- Paramètre de régulation `c`

### 🔬 Analyse :
- Profondeur d’arbre
- Qualité des actions sélectionnées
- Comparaison UCT vs Flat MCTS vs RL

---

## 📊 Évaluations & Analyses

| Méthode              | FrozenLake | CartPole | LunarLander |
|----------------------|------------|----------|-------------|
| Flat MCTS (Phase 1)  | ✅          | ✅        | ✅           |
| Q-Learning (Phase 2) | ✅          | ⚠️        | ❌           |
| MCTS amélioré (P3)   | ✅          | 🚫        | 🚫           |
| UCT MCTS (Phase 4)   | ✅          | ✅        | ✅           |

### 📈 Mesures :
- Taux de réussite
- Score moyen
- Temps de convergence
- Courbes de learning pour Q-learning

---

## 📝 Structure du rapport

1. **Introduction**
   - Motivation
   - Présentation des méthodes (MCTS, RL)

2. **Méthodes**
   - Flat MCTS
   - Q-Learning
   - MCTS optimisé
   - MCTS avec UCT

3. **Environnements**
   - Description et spécificités des jeux
   - Choix de paramètres

4. **Expériences**
   - Résultats détaillés
   - Graphiques (score, temps, convergence)
   - Rendus visuels ou captures

5. **Discussion**
   - Comparaisons croisées
   - Limites et efficacité
   - Cas où MCTS/RL fonctionne mieux

6. **Conclusion & perspectives**
   - Que faire ensuite ? (Deep RL, AlphaZero-like MCTS, généralisation)

---

## 🧠 Extensions possibles (bonus)
- Ajouter Mujoco (bonus ambitieux)
- Visualisation d’arbre MCTS (textuelle ou graphique)
- Analyse des politiques apprises
