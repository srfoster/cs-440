# cs-440

TODO:
* Programming env and lang?  https://colab.research.google.com/drive/1HeQsvxqIZkdb6tpIpkDW1Bfes9cmOZVN#scrollTo=4682551b
* Collab for the boring stuff and solara for cool sims?

Chapter 1: Intro to agents and classical search

Problems
- maze navigation
- n-queens
- 8-puzzle

Concepts
- agents, envs
- state spaces, search
- bfs/dfs/a*, cutoffs and deepening 


>> MAS?  Conway, boids, disease spread, polygon parable, sugar scape, tit for tat, etc...

Chapter 2: Beyond local search, optimization

Problems:
- hill climb (food seek); 8-queens; sim annealing, stochastic
- genetic algorithm; 8-queens
- [4.3] airport problem, local search, continuous space 

Concepts
- Objective function

Chapter 3: Adversarial search

Problem:
- tic tac toe, chess, etc..., a/b pruning
- stochastic game, expectiminimax

Concepts:
- Game theory, nash eqs?
- minimax strategy/alg  
- monte carlo sim

Chapter 3: constraint satisfaction... 

Problem:
- sudoku / map coloring.  (Needs to demo: search and inference relationship)

Concepts:
- vars, domains, constraints.  discrete vs continuous domains
- search vs inference
- inference: node, arc, path, and k-consistency
- complexity: The complexity of solving a CSP is strongly related to the structure of its constraint
graph. Tree-structured problems can be solved in linear time. Cutset conditioning can
reduce a general CSP to a tree-structured one and is quite efficient if a small cutset can
be found. Tree decomposition techniques transform the CSP into a tree of subproblems
and are efficient if the tree width of the constraint graph is small.

Chapter 4: Logic

Problem:
- unification
- forward/backward chaining

Chapter 5: Probabilistic reasoning

Bayesian networks
  Exact algorithms
  Stochastic approximation techniques such as likelihood weighting and Markov chain
  Monte Carlo can give reasonable estimates of the true posterior probabilities in a network and can cope with much larger networks than can exact algorithms

Reasoning over time
  hidden Markov models, Kalman filters, and dynamic Bayesian networks 
  particle filtering (approximation)

Chapter 6: Machine learning

Concepts: supervised, unsupervised, etc.

Neural nets: feedforward, back prop

Vision example.

Chapter 7: LLMs

"The transformer" architecture
Make your own ChatGPT