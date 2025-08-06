# Trame de réalisation du projet : Implémentation et optimisation de MCTS sur CliffWalking

---

## Introduction

- Présentation de l'environnement CliffWalking (Gymnasium)
- Objectif : Implémenter et optimiser un agent basé sur Monte Carlo Tree Search (MCTS)
- Phases du projet : progression de la méthode simple à la méthode optimisée

---

## Phase 1 : Implémentation du Flat MCTS

### Objectifs

- Comprendre les bases du Monte Carlo Tree Search
- Implémenter une version simple qui évalue chaque action par des simulations indépendantes (playouts aléatoires ou biaisés)
- Tester et analyser les performances initiales

### Étapes

- Implémenter la classe FlatMCTS avec méthode `random_playout`
- Implémenter la sélection d'action basée sur la moyenne des récompenses des playouts
- Évaluer l’agent sur plusieurs épisodes de CliffWalking
- Visualiser la politique apprise et les performances (scores, taux de succès, longueur des épisodes)

---

## Phase 2 : Implémentation de MCTS avec UCT

### Objectifs

- Ajouter la structure d’arbre avec nœuds stockant visites et valeurs cumulées
- Implémenter la sélection avec la formule UCT (Upper Confidence Bound applied to Trees)
- Ajouter les phases d’expansion, simulation (playout) et backpropagation
- Améliorer la qualité des décisions et la convergence vers une politique optimale

### Étapes

- Implémenter la classe MCTSNode pour représenter les nœuds de l’arbre
- Implémenter la classe MCTS avec méthodes : `tree_policy`, `expand`, `playout`, `backpropagate`, `best_action`
- Utiliser une politique biaisée dans les playouts pour améliorer la simulation
- Évaluer l’agent MCTS sur CliffWalking et comparer avec la phase 1
- Analyser les résultats et visualiser la politique

---

## Phase 3 : Optimisations avancées inspirées de la littérature

### Objectifs

- Intégrer des optimisations spécifiques pour maximiser la performance
- S’inspirer des techniques issues de papiers de recherche (ex : gestion fine des états, parallélisation, heuristiques spécifiques)
- Réduire le temps de calcul tout en améliorant la qualité des décisions

### Étapes

- Étudier les papiers de recherche pertinents sur MCTS et environnements similaires
- Implémenter des améliorations telles que :
  - Réutilisation efficace des environnements clonés
  - Politique de playout améliorée et adaptative
  - Parallélisation des simulations
  - Intégration d’heuristiques spécifiques à CliffWalking
- Tester et comparer les performances avec les phases précédentes
- Visualiser et analyser les résultats finaux

---

## Annexes

- Visualisation des politiques et des résultats (graphiques, heatmaps)
- Analyse des performances (temps de calcul, taux de succès, robustesse)
- Discussion sur les limites et perspectives d’amélioration

---

## Conclusion

- Synthèse des résultats obtenus à chaque phase
- Apports de chaque amélioration
- Perspectives pour des travaux futurs (ex : Deep MCTS, apprentissage par renforcement hybride)

---

## Références

- Articles et papiers de recherche utilisés
- Documentation Gymnasium et ressources MCTS
