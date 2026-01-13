// Auto-generated compiled content - DO NOT EDIT
// Generated on 2026-01-13T21:16:54.920Z

export const compiledFiles = {
  "content/chapter-01/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Intelligent Agents and Search",
          "concepts": [
            {
              "name": "Agents, Environments, and Search",
              "description": "Defining states, actions, and goals for search problems",
              "exam_questions": [
                "questions/astar-heuristic.yml",
                "questions/8puzzle-formulation.yml",
                "questions/search-algorithms.yml"
              ]
            }
          ]
        },
        {
          "category": "Local Search and Optimization",
          "concepts": [
            {
              "name": "Objective Functions, Optimization, and Local Search",
              "description": "Defining optimization criteria",
              "exam_questions": [
                "questions/8queens-objective.yml",
                "questions/8queens-hillclimb.yml",
                "questions/stochastic-optimization.yml"
              ]
            },
            {
              "name": "Continuous Optimization",
              "description": "Optimization in continuous spaces",
              "exam_questions": [
                "questions/airport-optimization.yml"
              ]
            }
          ]
        },
        {
          "category": "Adversarial Search and Games",
          "concepts": [
            {
              "name": "Minimax, Alpha-Beta, and Expectiminimax",
              "description": "Optimal play in two-player zero-sum games",
              "exam_questions": [
                "questions/minimax-alphabeta.yml",
                "questions/expectiminimax.yml",
                "questions/monte-carlo.yml",
                "questions/nash-equilibrium.yml"
              ]
            }
          ]
        }
      ]
    }
  },
  "content/chapter-01/questions/8puzzle-formulation.yml": {
    "type": "yaml",
    "module": {
      "id": "m1-8puzzle-formulation",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Solve the 8-puzzle problem by formulating it as a search problem.**\n\n**Level 0** Discuss the significance of state space formulation in AI and how it applies to the 8-puzzle problem.\n\n**Level 1** Describe the state space, actions, and goal test for the 8-puzzle problem, and give pseudocode for the functions ACTIONS(s), RESULT(s,a), and GOAL_TEST(s).  Demonstrate searching the state space by hand.  (You may assume states are represented as 3×3 lists of integers, where 0 represents the blank tile.)\n\n**Level 2** Implement a solution to the 8-puzzle problem in a language of your choice.\n\n**Level 3** Analyze the time and space complexity of your solution. Discuss how the choice of search algorithm (BFS, DFS, A*, etc.) affects performance and optimality.  Discuss the concept of heuristics and how they can be used to improve search performance in the context of the 8-puzzle problem.  How might things change in a larger puzzle (e.g., 15-puzzle) or a different grid size?\n",
      "answer": "The 8-puzzle requires three key functions: ACTIONS(s) returns valid moves based on the blank tile's position (UP, DOWN, LEFT, RIGHT within grid bounds), RESULT(s,a) creates a new state by swapping the blank with an adjacent tile, and GOAL_TEST(s) checks if the state matches the goal configuration [[1,2,3],[4,5,6],[7,8,0]]. This formulation enables any search algorithm to solve the puzzle.",
      "answer_kindergarten": "The 8-puzzle is like a sliding tile game. You have 8 numbered tiles and one empty space in a small square. You can slide tiles into the empty space. We need to tell the computer: which tiles can move right now, what happens when we move a tile, and when we've won the game by getting all the tiles in the right order.",
      "answer_3rd_grade": "To solve the 8-puzzle with a computer, we need three special instructions: 1) Check which way the empty space can move (up, down, left, or right, but not off the board), 2) Make a new puzzle by sliding a tile into the empty space, and 3) Check if all the numbers are in order from 1 to 8 with the empty space at the bottom right.",
      "answer_7th_grade": "The 8-puzzle problem formulation uses three functions. ACTIONS(s) finds where the blank (0) is located and returns which directions it can move without going outside the 3×3 grid. RESULT(s,a) creates a copy of the state, finds the blank, and swaps it with the neighboring tile in the specified direction. GOAL_TEST(s) compares the current state with the goal state [[1,2,3],[4,5,6],[7,8,0]]. These functions provide everything a search algorithm needs to find a solution.",
      "answer_high_school": "Problem formulation in the 8-puzzle requires precise function definitions. ACTIONS(s) locates the blank tile at position (r,c) and returns a list of valid moves: 'UP' if r>0, 'DOWN' if r<2, 'LEFT' if c>0, 'RIGHT' if c<2. RESULT(s,a) performs a deep copy of state s, calculates the new blank position based on action a, swaps the blank with the target tile, and returns the new state. GOAL_TEST(s) performs an equality check against the canonical goal configuration. This abstraction allows separation of problem definition from search strategy.",
      "answer_undergraduate": "The 8-puzzle exemplifies state-space search problem formulation. The state space consists of all possible 3×3 configurations with tiles {0,1,2,3,4,5,6,7,8}, though only 9!/2 = 181,440 states are reachable from any given initial state due to parity constraints. ACTIONS(s): Θ(1) operation computing valid moves from blank position. RESULT(s,a): Θ(n²) for grid copying, though optimizable with persistent data structures. GOAL_TEST(s): Θ(n²) equality check. The formulation's key insight is that by abstracting the problem into these three functions plus a cost function (uniform cost = 1 per move), we can apply any search algorithm (BFS, A*, IDA*, etc.) without modifying the search code itself.",
      "topics": [
        "Problem Formulation",
        "State Space",
        "Search Problems"
      ],
      "vocab_answer": [
        {
          "word": "state",
          "definition": "A complete description of the puzzle configuration at a particular moment"
        },
        {
          "word": "action",
          "definition": "A move or operation that transforms one state into another"
        },
        {
          "word": "goal test",
          "definition": "A function that determines whether a state is the desired solution"
        }
      ],
      "vocab_kindergarten": [
        {
          "word": "tile",
          "definition": "One of the numbered squares in the puzzle"
        },
        {
          "word": "empty space",
          "definition": "The spot where no tile is, where you can slide tiles into"
        }
      ],
      "vocab_3rd_grade": [
        {
          "word": "function",
          "definition": "A set of instructions that does a specific job"
        },
        {
          "word": "grid",
          "definition": "A pattern of squares arranged in rows and columns"
        }
      ],
      "vocab_7th_grade": [
        {
          "word": "deep copy",
          "definition": "Making a complete duplicate of data so changes don't affect the original"
        },
        {
          "word": "valid moves",
          "definition": "Actions that are allowed according to the rules of the problem"
        }
      ],
      "vocab_high_school": [
        {
          "word": "abstraction",
          "definition": "Hiding complex details behind simple interfaces"
        },
        {
          "word": "canonical",
          "definition": "The standard or official version of something"
        }
      ],
      "vocab_undergraduate": [
        {
          "word": "parity constraints",
          "definition": "Mathematical restrictions on which configurations can be reached from others"
        },
        {
          "word": "persistent data structures",
          "definition": "Data structures that preserve previous versions when modified"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=YUBvjEdBZZk",
        "https://www.youtube.com/watch?v=JYj-K4PuUCY"
      ]
    }
  },
  "content/chapter-01/questions/8queens-hillclimb.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-8queens-hillclimb",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Solve the 8-Queens problem using hill climbing optimization.**\n\n**Level 0** Discuss greedy local search and its limitations. Why do local minima, plateaus, and ridges pose problems? What makes hill climbing \"greedy\"?\n\n**Level 1** Using the 8-Queens formulation and CONFLICTS function from the previous question, describe the hill climbing algorithm. What defines a \"neighbor\" state? Give pseudocode for HILL_CLIMB(board) that moves to the best neighbor until no improvement is possible. Demonstrate by hand on a sample board.\n\n**Level 2** Implement hill climbing for 8-Queens. Include neighbor generation (moving each queen to each row in its column) and the greedy selection step. Run multiple random restarts and report success rate.\n\n**Level 3** Analyze the completeness and optimality of hill climbing. For 8-Queens, estimate the success rate from random starting positions (~86%). Discuss strategies to escape local optima: random restart, sideways moves, random walk. Compare time complexity O(n² × restarts) vs. backtracking O(n!).\n",
      "answer": "Hill climbing for 8-Queens uses a state representation where board[col]=row indicates queen positions. CONFLICTS counts pairs of queens attacking each other (same row or diagonal). The algorithm iteratively examines all neighbors (moving one queen in its column), selects the neighbor with fewest conflicts, and repeats until no improvement is possible. It's incomplete due to local optima but very fast when it succeeds.",
      "answer_kindergarten": "The 8-Queens problem is like trying to place 8 chess queens on a chessboard so none of them can capture each other. Hill climbing tries moving one queen at a time to a better spot, always picking the move that causes the fewest fights between queens. Sometimes it gets stuck because every move looks worse, even though there's a better solution if you keep trying.",
      "answer_3rd_grade": "In the 8-Queens puzzle, you need to place 8 queens on a chess board so no two queens can attack each other. We represent this by saying which row each queen is in for each column (like board[3]=5 means the queen in column 3 is in row 5). To solve it, we count how many queen-pairs are attacking (conflicts), then try moving each queen up or down in its column. We pick the move that reduces conflicts the most and keep going until we can't improve anymore.",
      "answer_7th_grade": "Hill climbing for 8-Queens starts with a random placement. CONFLICTS(board) counts attacking queen pairs by checking if board[i]==board[j] (same row) or abs(board[i]-board[j])==abs(i-j) (same diagonal) for all pairs. The algorithm generates all neighbors by trying each possible row for each column, evaluates conflicts for each neighbor, and moves to the best one. It stops when the current state has fewer conflicts than all neighbors (local optimum). Random restarts help escape local optima.",
      "answer_high_school": "State representation: board[0..7] where board[col]=row. Objective: minimize CONFLICTS(board) = number of attacking queen pairs. Algorithm: (1) Start with random board. (2) Generate all neighbors by moving each queen to each possible row in its column (7×8=56 neighbors). (3) Evaluate conflicts for each. (4) Move to best neighbor if better than current. (5) Repeat until no improvement. Complexity: O(n³) per iteration for n-Queens (n columns × n rows × O(n) conflict check). Gets stuck in local optima ~14% of the time for 8-Queens. Solutions: random restart (repeat with new random start), simulated annealing (accept worse moves probabilistically), or sideways moves (continue when plateau).",
      "answer_undergraduate": "Hill climbing is a local search algorithm navigating the complete-state formulation space. For N-Queens: State space size O(n^n), but gradient descent via steepest-ascent hill climbing evaluates O(n²) neighbors per step. CONFLICTS objective: O(n²) time via explicit pair enumeration, optimizable to O(n) with auxiliary data structures tracking attacks per row/diagonal. Completeness: No - guaranteed to reach local optimum, not global. Optimality: No - solution quality depends on topology of state-space landscape. For 8-Queens, ~86% success rate from random starts. Variants: Stochastic hill climbing (random among improving moves), first-choice (accept first improvement), random-restart (multiple independent runs). Theoretical analysis: Expected number of restarts = 1/p where p is basin of attraction for global optima. N-Queens has many global optima (92 solutions for N=8, with symmetries), increasing success probability. Local search trades completeness for speed: O(n² × restarts) vs. O(n!) for backtracking, highly effective for large N where systematic search is infeasible.",
      "topics": [
        "Local Search",
        "Hill Climbing",
        "Optimization",
        "N-Queens"
      ],
      "vocab_answer": [
        {
          "word": "local optimum",
          "definition": "A state better than all its neighbors but not the best overall solution"
        },
        {
          "word": "objective function",
          "definition": "A function that measures the quality of a state"
        },
        {
          "word": "neighbor",
          "definition": "A state that can be reached by a single small modification"
        }
      ],
      "vocab_kindergarten": [
        {
          "word": "attack",
          "definition": "When two chess queens can capture each other"
        },
        {
          "word": "stuck",
          "definition": "When you can't make progress anymore"
        }
      ],
      "vocab_3rd_grade": [
        {
          "word": "conflicts",
          "definition": "Problems or fights between pieces"
        },
        {
          "word": "improve",
          "definition": "To make something better"
        }
      ],
      "vocab_7th_grade": [
        {
          "word": "random restart",
          "definition": "Starting over from a new random position when stuck"
        },
        {
          "word": "evaluate",
          "definition": "To calculate or measure the quality of something"
        }
      ],
      "vocab_high_school": [
        {
          "word": "plateau",
          "definition": "A flat region in the search space where neighbors have equal value"
        },
        {
          "word": "complexity",
          "definition": "How much time or space an algorithm requires"
        }
      ],
      "vocab_undergraduate": [
        {
          "word": "basin of attraction",
          "definition": "The set of states from which local search reaches a particular local optimum"
        },
        {
          "word": "state-space landscape",
          "definition": "The topology formed by states and their objective values"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=lgqI9M94eKo",
        "https://www.youtube.com/watch?v=oSKNx6E0hPE"
      ]
    }
  },
  "content/chapter-01/questions/8queens-objective.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-8queens-objective",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Formulate the 8-Queens problem as an optimization problem and implement the CONFLICTS objective function.**\n\n**Level 0** Discuss the difference between path-finding search (Chapter 1) and optimization-based search (Chapter 2). Why is local search appropriate for problems like N-Queens? What are the trade-offs?\n\n**Level 1** Describe the 8-Queens problem as an optimization problem. Using the representation board[col]=row, explain what constitutes a conflict between queens. Give pseudocode for CONFLICTS(board) that counts attacking queen pairs. Demonstrate by hand on a sample board.\n\n**Level 2** Implement the CONFLICTS function in a language of your choice. Generate random 8-Queens boards and verify your conflict counting is correct.\n\n**Level 3** Analyze the time complexity of CONFLICTS(board). Discuss how the objective function landscape affects search: are there many local optima? How does the number of global optima (92 for 8-Queens with symmetries) relate to search difficulty?\n",
      "answer": "The CONFLICTS function counts pairs of queens that attack each other either on the same row or the same diagonal. For N-Queens represented as board[col]=row, two queens at positions (i, board[i]) and (j, board[j]) conflict if they're on the same row or same diagonal.\n\nPseudocode:\n```\nfunction CONFLICTS(board):\n    n = len(board)\n    c = 0\n    for i in 0..n-1:\n        for j in i+1..n-1:\n            sameRow = (board[i] == board[j])\n            sameDiag = (abs(board[i]-board[j]) == abs(i-j))\n            if sameRow or sameDiag:\n                c += 1\n    return c\n```\n",
      "topics": [
        "N-Queens",
        "Objective Functions",
        "Optimization"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=xouin83ebxE"
      ]
    }
  },
  "content/chapter-01/questions/airport-optimization.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-airport-optimization",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Solve the airport placement problem using continuous optimization.**\n\n**Level 0** Discuss the difference between discrete optimization (8-Queens) and continuous optimization (airport placement). When is \"state space as a graph\" not the right model? Why don't discrete search methods work well here?\n\n**Level 1** Describe the airport placement problem: find (x,y) that minimizes weighted sum of squared distances to towns. Give pseudocode for AIRPORT_OBJECTIVE(x,y,towns) and RANDOM_SEARCH_CONTINUOUS(towns, bounds, iters). Demonstrate by hand with 3 towns.\n\n**Level 2** Implement both functions. Compare random search with gradient descent (compute partial derivatives ∂f/∂x and ∂f/∂y analytically). Visualize the objective function as a heat map.\n\n**Level 3** Prove that the optimal solution is the weighted centroid: x* = Σ(w_i * x_i)/Σw_i. Analyze why this problem has a single global optimum (convex function). Discuss how this differs from the 8-Queens landscape. Generalize to other continuous optimization problems: when can we use calculus vs. metaheuristics?\n",
      "answer": "This is a continuous optimization problem where we want to find the best location for an airport that minimizes weighted distances to multiple towns. Random search samples many random points and keeps the best.\n\nPseudocode:\n```\nfunction AIRPORT_OBJECTIVE(x, y, towns):\n    # towns: list of (tx,ty, weight)\n    s = 0\n    for (tx,ty,w) in towns:\n        dx = x - tx; dy = y - ty\n        s += w * (dx*dx + dy*dy)\n    return s\n\nfunction RANDOM_SEARCH_CONTINUOUS(towns, bounds, iters):\n    best = null\n    bestVal = +infinity\n    for i in 1..iters:\n        x = randomUniform(bounds.xMin, bounds.xMax)\n        y = randomUniform(bounds.yMin, bounds.yMax)\n        v = AIRPORT_OBJECTIVE(x,y,towns)\n        if v < bestVal:\n            bestVal = v\n            best = (x,y)\n    return best\n```\n",
      "topics": [
        "Continuous Optimization",
        "Random Search",
        "Facility Location"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=IHZwWFHWa-w"
      ]
    }
  },
  "content/chapter-01/questions/astar-heuristic.yml": {
    "type": "yaml",
    "module": {
      "id": "m1-astar-heuristic",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Implement the A* search algorithm -- A_STAR(start, h) -- for maze navigation using a priority queue ordered by f=g+h. Your implementation should maintain g-costs and support \"decrease-key\" behavior when a better path to a state is found.**\n\n**Level 0** Discuss the significance of search in the context of AI.  Discuss A* search and how it differs from uninformed search algorithms like BFS and DFS.\n\n**Level 1** For a sample maze, demonstrate A* search showing the priority queue and distance updates. Give pseudocode.\n\n**Level 2** Implement A* search in a language of your choice, including path reconstruction.\n\n**Level 3** Analyze the time and space complexity of A* search. Explain how the choice of heuristic h affects performance and optimality. Discuss admissible heuristics and give examples.\n",
      "answer": "A* search uses f(n) = g(n) + h(n) where g(n) is the actual cost from start to n and h(n) is the estimated cost from n to goal. It maintains a priority queue (open list) ordered by f-value. A* is optimal when h is admissible (never overestimates). Better heuristics (higher h while staying admissible) explore fewer nodes. Manhattan distance is a classic admissible heuristic for grid-based problems.",
      "answer_undergraduate": "A* optimally solves shortest-path problems when h is admissible. Proof of optimality: Suppose suboptimal goal G2 with f(G2) < f(G1) where G1 is optimal. Let n be an unexpanded node on optimal path to G1. Then f(n) = g(n) + h(n) ≤ g(n) + h*(n) = f*(n) = C* (optimal cost). But f(G2) = g(G2) > C*, contradicting f(G2) < f(G1) = C*. Time/space: O(b^d) in worst case, but effective branching factor depends on h accuracy. A* with consistent h expands nodes in order of increasing g-value, equivalent to Dijkstra's on an implicit graph. Heuristic quality measured by effective branching factor b*: N = 1 + b* + (b*)^2 + ... + (b*)^d where N is nodes expanded. Well-designed heuristics can reduce b* dramatically. Pattern databases, linear conflict, and disjoint pattern additivity are advanced techniques for creating powerful admissible heuristics.",
      "topics": [
        "A* Search",
        "Heuristic Functions",
        "Informed Search",
        "Optimality"
      ],
      "vocab_answer": [
        {
          "word": "admissible",
          "definition": "A heuristic that never overestimates the true cost to reach the goal"
        },
        {
          "word": "priority queue",
          "definition": "A data structure that returns elements in order of priority (typically minimum value first)"
        },
        {
          "word": "Manhattan distance",
          "definition": "The sum of horizontal and vertical distances, named after the grid layout of Manhattan"
        }
      ],
      "vocab_kindergarten": [
        {
          "word": "guess",
          "definition": "To think about what might happen without knowing for sure"
        },
        {
          "word": "total",
          "definition": "The amount you get when you add things together"
        }
      ],
      "vocab_3rd_grade": [
        {
          "word": "priority",
          "definition": "Which things are more important to do first"
        },
        {
          "word": "estimate",
          "definition": "A smart guess about a number or amount"
        }
      ],
      "vocab_7th_grade": [
        {
          "word": "successor",
          "definition": "A state that can be reached directly from the current state"
        },
        {
          "word": "extract",
          "definition": "To take out or remove (in this case, from a priority queue)"
        }
      ],
      "vocab_high_school": [
        {
          "word": "consistent",
          "definition": "A heuristic where h(n) ≤ cost(n,n') + h(n') for any edge"
        },
        {
          "word": "optimal path",
          "definition": "The shortest or lowest-cost path from start to goal"
        }
      ],
      "vocab_undergraduate": [
        {
          "word": "effective branching factor",
          "definition": "A measure of search efficiency: the average number of nodes generated per level"
        },
        {
          "word": "pattern database",
          "definition": "Precomputed exact costs for subproblems used as heuristic values"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=ySN5Wnu88nE",
        "https://www.youtube.com/watch?v=6TsL96NAZCo",
        "https://www.youtube.com/watch?v=71CEj4gKDnE"
      ]
    }
  },
  "content/chapter-01/questions/expectiminimax.yml": {
    "type": "yaml",
    "module": {
      "id": "m3-expectiminimax",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Extend minimax to stochastic games using Expectiminimax.**\n\n**Level 0** Discuss how randomness changes adversarial reasoning. Why can't we just use minimax for games with dice? What does \"expected utility\" mean in this context? Give examples of games that require expectiminimax (backgammon, Monopoly, etc.).\n\n**Level 1** Describe expectiminimax: MAX nodes (your turn), MIN nodes (opponent's turn), and CHANCE nodes (dice/cards). At CHANCE nodes, compute expected value using outcome probabilities. Give pseudocode for EXPECTIMINIMAX(s, player) that branches on node type and uses OUTCOMES(s) to get (outcome, probability) pairs. Demonstrate by hand on a simple dice game tree.\n\n**Level 2** Implement expectiminimax for a simple dice game (e.g., a game where players choose to roll 1 or 2 dice and highest total wins). Compare decisions with and without considering probabilities.\n\n**Level 3** Analyze why alpha-beta pruning doesn't directly apply to expectiminimax (averaging breaks the minimax property). Discuss bounded expectiminimax and Monte Carlo approximations. How does the branching factor from chance nodes affect tractability? Compare expectiminimax complexity with minimax.\n",
      "answer": "Expectiminimax extends minimax to handle games with randomness (like dice rolls). CHANCE nodes compute expected values over possible outcomes weighted by their probabilities, while MAX and MIN nodes work as in regular minimax.\n\nPseudocode:\n```\nfunction EXPECTIMINIMAX(s, player):\n    if TERMINAL(s): return UTILITY(s, player)\n    if NODE_TYPE(s) == \"MAX\":\n        v = -infinity\n        for a in ACTIONS(s):\n            v = max(v, EXPECTIMINIMAX(RESULT(s,a), player))\n        return v\n    if NODE_TYPE(s) == \"MIN\":\n        v = +infinity\n        for a in ACTIONS(s):\n            v = min(v, EXPECTIMINIMAX(RESULT(s,a), player))\n        return v\n    if NODE_TYPE(s) == \"CHANCE\":\n        v = 0\n        for (outcome, p) in OUTCOMES(s):\n            v += p * EXPECTIMINIMAX(outcome, player)\n        return v\n```\n",
      "topics": [
        "Expectiminimax",
        "Stochastic Games",
        "Game Theory"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=jaFRyzp7yWw"
      ]
    }
  },
  "content/chapter-01/questions/minimax-alphabeta.yml": {
    "type": "yaml",
    "module": {
      "id": "m3-minimax-alphabeta",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Implement optimal adversarial search using Minimax and Alpha-Beta pruning.**\n\n**Level 0** Discuss the shift from single-agent search (Chapter 1) to adversarial search. Why does having an opponent fundamentally change the problem? What makes a game \"zero-sum\"? Why is \"assume optimal opponent\" a reasonable design choice?\n\n**Level 1** Describe the minimax algorithm for deterministic two-player zero-sum games. How do MAX and MIN nodes alternate? How does utility propagate up the tree? Give pseudocode for MINIMAX_DECISION(state, player) and recursive helpers MAX_VALUE and MIN_VALUE using TERMINAL, UTILITY, ACTIONS, and RESULT. Demonstrate by hand on a tic-tac-toe subtree.\n\nThen describe alpha-beta pruning. What do alpha and beta represent? When can we prune? Give pseudocode for ALPHABETA_DECISION with AB_MAX and AB_MIN. Demonstrate pruning by hand on a game tree.\n\n**Level 2** Implement both minimax and alpha-beta for tic-tac-toe or another simple game. Compare the number of nodes expanded. Test with different move orderings to see pruning effectiveness.\n\n**Level 3** Prove that alpha-beta returns the same value as minimax (correctness). Analyze complexity: minimax is O(b^d), but with optimal move ordering alpha-beta achieves O(b^(d/2)). Discuss why move ordering matters and strategies like killer moves, history heuristic, and iterative deepening. When do we need depth limits and evaluation functions?\n",
      "answer": "Minimax computes optimal play by recursively evaluating MAX (maximizer's turn) and MIN (minimizer's turn) layers. Alpha-beta pruning eliminates branches that cannot affect the final decision by maintaining alpha (best MAX option) and beta (best MIN option) bounds. If beta ≤ alpha, remaining siblings can be pruned. Optimal move ordering achieves O(b^(d/2)) vs. O(b^d) for minimax.",
      "topics": [
        "Minimax",
        "Alpha-Beta Pruning",
        "Game Trees"
      ],
      "vocab_answer": [
        {
          "word": "adversarial",
          "definition": "Involving opponents with conflicting goals"
        },
        {
          "word": "utility",
          "definition": "The payoff or value of a game outcome"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=l-hh51ncgDI"
      ]
    }
  },
  "content/chapter-01/questions/monte-carlo.yml": {
    "type": "yaml",
    "module": {
      "id": "m3-monte-carlo",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Use Monte Carlo Tree Search when exact minimax is intractable.**\n\n**Level 0** Discuss the limitations of minimax/alpha-beta for large games. Why is Go harder than Chess? How does Monte Carlo simulation provide an alternative to exhaustive search? What's the key insight: \"playing randomly is informative\"?\n\n**Level 1** Describe simple Monte Carlo move evaluation using random playouts. For each legal move, simulate N random games from the resulting position and track win rate. Give pseudocode for MONTE_CARLO_MOVE(state, player, rolloutsPerMove) and RANDOM_PLAYOUT(state, player). Demonstrate by hand with a simple game.\n\nThen describe full MCTS with four phases: **Selection** (traverse tree using UCB), **Expansion** (add new node), **Simulation** (random playout), **Backpropagation** (update statistics). Explain the UCB1 formula: score = wins/visits + C * sqrt(ln(parent_visits)/visits).\n\n**Level 2** Implement simple Monte Carlo move selection with random rollouts for tic-tac-toe or connect-four. Then implement full MCTS with UCB selection. Compare performance and number of simulations needed.\n\n**Level 3** Analyze the exploration-exploitation trade-off in UCB1. Prove that UCB converges to optimal move selection with enough samples. Discuss why MCTS excels in high-branching games (Go has b≈250 vs Chess b≈35). Compare MCTS with alpha-beta: when is each preferable? Discuss neural network enhancements (AlphaGo).\n",
      "answer": "Monte Carlo move selection evaluates each legal move by performing random playouts from the resulting state. For each move, run N simulations playing randomly until game end, record wins/losses, and choose the move with highest win rate. MCTS extends this with tree search, UCB selection, and incremental tree building.",
      "topics": [
        "Monte Carlo Methods",
        "MCTS",
        "Stochastic Search"
      ],
      "vocab_answer": [
        {
          "word": "rollout",
          "definition": "A simulated game played with random moves from a given state"
        },
        {
          "word": "UCB",
          "definition": "Upper Confidence Bound - balances exploitation and exploration"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=UXW2yZndl7U"
      ]
    }
  },
  "content/chapter-01/questions/nash-equilibrium.yml": {
    "type": "yaml",
    "module": {
      "id": "m3-nash-equilibrium",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Find Nash equilibria in general-sum games beyond zero-sum adversarial search.**\n\n**Level 0** Discuss how Nash equilibrium generalizes minimax reasoning to non-zero-sum games. Why is \"assume optimal opponent\" not enough when interests aren't perfectly opposed? What makes Nash equilibrium a \"stable\" solution concept? Give examples: Prisoner's Dilemma, Battle of the Sexes, Rock-Paper-Scissors.\n\n**Level 1** Define Nash equilibrium: a strategy profile where no player can improve by unilaterally deviating. Explain best-response reasoning. For 2×2 normal-form games with payoff matrices A (row player) and B (column player), describe how to find pure Nash equilibria by checking if each strategy pair (i,j) is a best response for both players. Give pseudocode for PURE_NASH_2x2(A, B). Demonstrate by hand on Prisoner's Dilemma and Matching Pennies.\n\n**Level 2** Implement PURE_NASH_2x2 and test on classic games. Explain why some games have no pure Nash equilibrium (Matching Pennies, Rock-Paper-Scissors). Introduce mixed strategies: probability distributions over actions. Compute a mixed Nash equilibrium by hand for a 2×2 game.\n\n**Level 3** Prove Nash's existence theorem: every finite game has at least one Nash equilibrium (possibly mixed). Discuss computational complexity: finding Nash equilibria is PPAD-complete. Compare Nash equilibrium with minimax value in zero-sum games. Discuss limitations: multiple equilibria, equilibrium selection, irrationality. How do Nash equilibria relate to multi-agent reinforcement learning?\n",
      "answer": "A pure Nash equilibrium is a strategy profile where no player can improve their payoff by unilaterally changing strategy. Check each strategy pair (i,j) to see if it's a best response for both players.\n\nPseudocode:\n```\nfunction PURE_NASH_2x2(A, B):\n    equilibria = []\n    for i in 0..1:          # row action\n        for j in 0..1:      # col action\n            # is (i,j) best response for row?\n            rowBest = true\n            for i2 in 0..1:\n                if A[i2][j] > A[i][j]:\n                    rowBest = false\n            # is (i,j) best response for col?\n            colBest = true\n            for j2 in 0..1:\n                if B[i][j2] > B[i][j]:\n                    colBest = false\n            if rowBest and colBest:\n                add (i,j) to equilibria\n    return equilibria\n```\n",
      "topics": [
        "Nash Equilibrium",
        "Game Theory",
        "Strategic Reasoning"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=NSVmOC_5zrE"
      ]
    }
  },
  "content/chapter-01/questions/search-algorithms.yml": {
    "type": "yaml",
    "module": {
      "id": "m1-search-algorithms",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Solve the 8-puzzle problem using uninformed and informed search algorithms.**\n\n**Level 0** Discuss the significance of search algorithms in AI and the fundamental tradeoffs between time, space, optimality, and completeness. How does adding domain knowledge (heuristics) change search performance?\n\n**Level 1** For the 8-puzzle problem (using ACTIONS, RESULT, and GOAL_TEST from the previous question), describe and give pseudocode for: (1) BFS - breadth-first search that avoids revisiting states, (2) DLS - depth-limited search that stops expanding beyond a depth limit, (3) IDDFS - iterative deepening that calls DLS with increasing limits, and (4) A* - informed search using f(n)=g(n)+h(n) with Manhattan distance heuristic. Demonstrate each algorithm by hand on a sample 8-puzzle instance.\n\n**Level 2** Implement all four search algorithms in a language of your choice. Include path reconstruction and compare their performance on multiple 8-puzzle instances.\n\n**Level 3** Analyze the time and space complexity of each algorithm. Discuss when each is preferred: BFS vs IDDFS for uninformed search, and how A* with Manhattan distance dramatically outperforms both. Explain admissibility and why Manhattan distance never overestimates. How would performance change for the 15-puzzle?  What other search algorithms could be applied to this problem, and how would they compare in terms of performance and optimality?\n",
      "answer": "BFS uses a queue (FIFO) to explore states level by level, guaranteeing optimal solutions but requiring O(b^d) space. DLS performs DFS but stops at a depth limit. IDDFS repeatedly runs DLS with increasing limits (0,1,2,...), achieving BFS's optimality with only O(bd) space. A* uses f(n)=g(n)+h(n) with a priority queue, where Manhattan distance sums each tile's distance from its goal position. For hard 8-puzzle instances: BFS/IDDFS explore ~170K states, A* with Manhattan distance explores ~1-5K states—a 30-100x improvement.",
      "answer_kindergarten": "Imagine solving a sliding tile puzzle. BFS is like trying every possible move one step at a time, then every two-step combination—you'll find the shortest solution but need to remember everything! IDDFS is smarter: try 1 move, then start over and try 2 moves, then 3 moves. A* is the smartest: it looks at how far each tile is from where it should be and focuses on moves that seem most promising first!",
      "answer_3rd_grade": "For the 8-puzzle: BFS explores all possible moves systematically—1 move away, 2 moves, 3 moves, etc. It finds the shortest solution but uses lots of memory. IDDFS searches depth 1, restarts and searches depth 2, then 3, saving memory. A* is clever: it adds up two numbers for each puzzle state: how many moves you've made so far, plus a guess of how many moves are left (by counting how far each tile is from its goal spot). It always explores the state with the smallest total first, finding solutions much faster!",
      "answer_7th_grade": "BFS maintains a queue of board states, explores level by level, and reconstructs the path using parent pointers. IDDFS runs depth-limited search repeatedly with increasing limits. A* maintains a priority queue ordered by f(n) = g(n) + h(n), where g(n) is moves made so far and h(n) is the Manhattan distance heuristic (sum of each tile's horizontal and vertical distance from its goal position). A* always expands the most promising state first. For the 8-puzzle, A* is dramatically faster because Manhattan distance provides good guidance toward the goal.",
      "answer_high_school": "BFS: FIFO queue, visited set, parent map for path reconstruction. O(b^d) time and space. IDDFS: Repeatedly calls DLS(depth) for depth=0,1,2,... DLS uses recursion with depth parameter. O(b^d) time, O(bd) space—much better than BFS. A*: Priority queue ordered by f=g+h. For 8-puzzle, Manhattan distance h(s) = Σ |current_row - goal_row| + |current_col - goal_col| for each tile. Manhattan distance is admissible (never overestimates) because each tile must move at least that many steps. A* expands far fewer nodes than BFS/IDDFS: typically 1K-5K states vs 170K states for hard instances.",
      "answer_undergraduate": "BFS: Complete and optimal for unit costs. Time/space O(b^d) where b≈2-3 for 8-puzzle (average branching factor), d=solution depth. For 8-puzzle, ~50% of random instances have d≤20. Space is the limiting factor. IDDFS: Time O(b^d), space O(bd). Revisits states but last level dominates: overhead = O(b^d / (b-1)) ≈ O(b^d). Preferred when memory-constrained. A*: With admissible h, expands all nodes with f(n) < C* and some with f(n)=C* (C*=optimal cost). Manhattan distance for 8-puzzle is admissible and consistent. Effective branching factor b*: For A* with Manhattan, typically b*≈1.2-1.5 vs b≈2.5 for IDDFS, yielding 30-100x fewer expansions. Time/space still O(b^d) worst-case, but typical performance is dramatically better. For 15-puzzle (d≈50-80), only A* variants (IDA*, pattern databases) are practical.",
      "topics": [
        "Search Algorithms",
        "BFS",
        "Depth-Limited Search",
        "IDDFS",
        "A* Search",
        "Heuristic Functions",
        "Manhattan Distance",
        "Completeness and Optimality"
      ],
      "vocab_answer": [
        {
          "word": "queue",
          "definition": "A first-in-first-out (FIFO) data structure"
        },
        {
          "word": "priority queue",
          "definition": "A data structure that returns elements in order of priority (minimum f-value first for A*)"
        },
        {
          "word": "heuristic",
          "definition": "An estimate or educated guess used to guide search"
        },
        {
          "word": "admissible",
          "definition": "A heuristic that never overestimates the true cost to reach the goal"
        },
        {
          "word": "Manhattan distance",
          "definition": "Sum of horizontal and vertical distances; each tile's displacement from its goal position"
        }
      ],
      "vocab_kindergarten": [
        {
          "word": "level",
          "definition": "How many steps away something is from where you started"
        },
        {
          "word": "remember",
          "definition": "To keep information about what you've already seen"
        }
      ],
      "vocab_3rd_grade": [
        {
          "word": "queue",
          "definition": "A waiting line where the first person in is the first person out"
        },
        {
          "word": "restart",
          "definition": "To go back to the beginning and start over"
        }
      ],
      "vocab_7th_grade": [
        {
          "word": "recursion",
          "definition": "When a function calls itself to solve smaller versions of a problem"
        },
        {
          "word": "parent pointer",
          "definition": "A reference to the previous state that led to the current state"
        }
      ],
      "vocab_high_school": [
        {
          "word": "FIFO",
          "definition": "First In, First Out - a queue discipline"
        },
        {
          "word": "backtracking",
          "definition": "Retracing steps from goal to start using stored parent information"
        }
      ],
      "vocab_undergraduate": [
        {
          "word": "branching factor",
          "definition": "Average number of successors per state"
        },
        {
          "word": "asymptotic complexity",
          "definition": "Behavior of an algorithm as input size approaches infinity"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=oDqjPvD54Ss",
        "https://www.youtube.com/watch?v=NUgMa5coCoE"
      ]
    }
  },
  "content/chapter-01/questions/simulated-annealing.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-stochastic-optimization",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Solve the 8-Queens problem using stochastic optimization: simulated annealing and genetic algorithms.**\n\n**Level 0** Discuss how randomness helps overcome the limitations of greedy hill climbing. Compare two paradigms: single-point stochastic search (simulated annealing) vs. population-based search (genetic algorithms). Why is the exploration-exploitation trade-off central to both?\n\n**Level 1** Using the 8-Queens formulation:\n\n**Simulated Annealing:** Describe the metallurgical annealing metaphor. Give pseudocode for SIM_ANNEAL(board, T0, alpha, steps) that accepts improvements always, and worse moves with probability exp(-delta/T). Demonstrate the acceptance probability calculation by hand.\n\n**Genetic Algorithm:** Describe the evolutionary components. How is a board represented as a \"chromosome\"? What is the fitness function? Describe selection, crossover, and mutation. Give pseudocode for GA_8QUEENS(popSize, generations, mutRate). Demonstrate one generation by hand with a small population.\n\n**Level 2** Implement both algorithms for 8-Queens:\n\n**SA Implementation:** Use geometric cooling schedule (T = alpha * T) with RANDOM_NEIGHBOR function. Test various temperature schedules.\n\n**GA Implementation:** Use tournament selection (k=3), one-point crossover, and mutation. Track population diversity over generations.\n\nCompare both with hill climbing: success rates, iterations to solution, and robustness.\n\n**Level 3** Analyze theoretical guarantees and practical trade-offs:\n\n**Simulated Annealing:** Prove that logarithmic cooling converges to global optimum with probability 1. Why use faster geometric cooling in practice? Analyze temperature's role in exploration vs. exploitation.\n\n**Genetic Algorithms:** Discuss the schema theorem (building blocks hypothesis). How do population size, mutation rate, and selection pressure affect convergence and diversity? Why might GAs outperform single-point search for certain landscapes?\n\nCompare when to use each: SA for well-defined neighborhoods, GAs for complex recombination, both for escaping local optima.\n",
      "answer": "**Simulated annealing** accepts worse moves with probability exp(-delta/T) where temperature T decreases over time (T = alpha * T). This escapes local optima early while converging as T→0. **Genetic algorithms** maintain a population, select parents by fitness, create offspring through crossover and mutation. For 8-Queens: fitness = 1/(1+conflicts), tournament selection picks best of k random, one-point crossover combines board segments, mutation randomly changes positions. Both overcome hill climbing's greedy limitations through controlled randomness.",
      "topics": [
        "Simulated Annealing",
        "Genetic Algorithms",
        "Stochastic Search",
        "Evolutionary Computation",
        "Population-based Search"
      ],
      "vocab_answer": [
        {
          "word": "temperature",
          "definition": "A parameter controlling randomness in simulated annealing"
        },
        {
          "word": "cooling schedule",
          "definition": "The rate at which temperature decreases over time"
        },
        {
          "word": "crossover",
          "definition": "Combining genetic material from two parents to create offspring"
        },
        {
          "word": "mutation",
          "definition": "Random changes to maintain diversity in the population"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=eBmU1ONJ-os"
      ]
    }
  },
  "content/chapter-01/questions/stochastic-optimization.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-stochastic-optimization",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Solve the 8-Queens problem using stochastic optimization: simulated annealing and genetic algorithms.**\n\n**Level 0** Discuss how randomness helps overcome the limitations of greedy hill climbing. Compare two paradigms: single-point stochastic search (simulated annealing) vs. population-based search (genetic algorithms). Why is the exploration-exploitation trade-off central to both?\n\n**Level 1** Using the 8-Queens formulation:\n\n**Simulated Annealing:** Describe the metallurgical annealing metaphor. Give pseudocode for SIM_ANNEAL(board, T0, alpha, steps) that accepts improvements always, and worse moves with probability exp(-delta/T). Demonstrate the acceptance probability calculation by hand.\n\n**Genetic Algorithm:** Describe the evolutionary components. How is a board represented as a \"chromosome\"? What is the fitness function? Describe selection, crossover, and mutation. Give pseudocode for GA_8QUEENS(popSize, generations, mutRate). Demonstrate one generation by hand with a small population.\n\n**Level 2** Implement both algorithms for 8-Queens:\n\n**SA Implementation:** Use geometric cooling schedule (T = alpha * T) with RANDOM_NEIGHBOR function. Test various temperature schedules.\n\n**GA Implementation:** Use tournament selection (k=3), one-point crossover, and mutation. Track population diversity over generations.\n\nCompare both with hill climbing: success rates, iterations to solution, and robustness.\n\n**Level 3** Analyze theoretical guarantees and practical trade-offs:\n\n**Simulated Annealing:** Prove that logarithmic cooling converges to global optimum with probability 1. Why use faster geometric cooling in practice? Analyze temperature's role in exploration vs. exploitation.\n\n**Genetic Algorithms:** Discuss the schema theorem (building blocks hypothesis). How do population size, mutation rate, and selection pressure affect convergence and diversity? Why might GAs outperform single-point search for certain landscapes?\n\nCompare when to use each: SA for well-defined neighborhoods, GAs for complex recombination, both for escaping local optima.\n",
      "answer": "**Simulated annealing** accepts worse moves with probability exp(-delta/T) where temperature T decreases over time (T = alpha * T). This escapes local optima early while converging as T→0. **Genetic algorithms** maintain a population, select parents by fitness, create offspring through crossover and mutation. For 8-Queens: fitness = 1/(1+conflicts), tournament selection picks best of k random, one-point crossover combines board segments, mutation randomly changes positions. Both overcome hill climbing's greedy limitations through controlled randomness.",
      "topics": [
        "Simulated Annealing",
        "Genetic Algorithms",
        "Stochastic Search",
        "Evolutionary Computation",
        "Population-based Search"
      ],
      "vocab_answer": [
        {
          "word": "temperature",
          "definition": "A parameter controlling randomness in simulated annealing"
        },
        {
          "word": "cooling schedule",
          "definition": "The rate at which temperature decreases over time"
        },
        {
          "word": "crossover",
          "definition": "Combining genetic material from two parents to create offspring"
        },
        {
          "word": "mutation",
          "definition": "Random changes to maintain diversity in the population"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=eBmU1ONJ-os"
      ]
    }
  },
  "content/chapter-02/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Constraint Satisfaction Problems",
          "concepts": [
            {
              "name": "CSP Representation and Search",
              "description": "Variables, domains, constraints, and backtracking",
              "exam_questions": [
                "questions/csp-fundamentals.yml",
                "questions/ac3-algorithm.yml",
                "questions/csp-structure.yml"
              ]
            }
          ]
        },
        {
          "category": "Logic and Inference",
          "concepts": [
            {
              "name": "Unification, Inference, Resolution, and Search",
              "description": "Pattern matching for logical terms",
              "exam_questions": [
                "questions/unification.yml",
                "questions/horn-clause-inference.yml",
                "questions/resolution.yml"
              ]
            }
          ]
        }
      ]
    }
  },
  "content/chapter-02/questions/ac3-algorithm.yml": {
    "type": "yaml",
    "module": {
      "id": "m4-ac3-algorithm",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Enforce arc consistency using the AC-3 algorithm.**\n\n**Level 0** Discuss why forward checking isn't enough: it only looks one step ahead. What is arc consistency? Why is it more powerful than forward checking but still incomplete (doesn't solve the CSP)? When should you use AC-3: as preprocessing, during search, or both?\n\n**Level 1** Define arc consistency: for arc (X,Y), every value x in D_X must have some value y in D_Y that satisfies the constraint between X and Y. Describe AC-3: maintain queue of arcs, repeatedly call REVISE(X,Y) which removes unsupported values from D_X. When D_X changes, add all arcs (Z,X) back to queue. Give pseudocode for AC3(csp) and REVISE(csp, X, Y). Demonstrate by hand on a map-coloring problem.\n\n**Level 2** Implement AC-3 with REVISE for binary not-equal constraints (map coloring). Test as preprocessing before backtracking. Compare: backtracking alone vs. AC-3 + backtracking vs. maintaining arc consistency (MAC) during search.\n\n**Level 3** Prove AC-3 terminates (domains only shrink, finite values). Analyze time complexity: O(cd³) where c=number of constraints, d=domain size. In practice, often O(ed²) where e=edges. Discuss why AC-3 doesn't solve all CSPs (example: 3-coloring a triangle is arc-consistent but has no solution). Introduce stronger notions: path consistency, k-consistency, and the trade-off between preprocessing cost and search reduction.\n",
      "answer": "AC-3 makes CSP arc-consistent: for every arc (X,Y), every value x in X's domain has some value y in Y's domain satisfying the constraint. Algorithm maintains queue of arcs, repeatedly calling REVISE(X,Y) which removes values from X's domain that have no supporting value in Y. When X's domain changes, add arcs (Z,X) back to queue. Returns false if any domain becomes empty.",
      "topics": [
        "Arc Consistency",
        "Constraint Propagation",
        "AC-3"
      ],
      "vocab_answer": [
        {
          "word": "arc consistent",
          "definition": "Property where every domain value has a compatible value in constrained neighbors"
        },
        {
          "word": "revise",
          "definition": "To update a domain by removing unsupported values"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=4cCS8rrYT14"
      ]
    }
  },
  "content/chapter-02/questions/csp-fundamentals.yml": {
    "type": "yaml",
    "module": {
      "id": "m4-csp-fundamentals",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Formulate and solve Constraint Satisfaction Problems using backtracking search with forward checking.**\n\n**Level 0** Discuss how CSPs differ from generic search problems (Chapter 1). Why is exploiting constraints more powerful than blind search? What makes problems like Sudoku, map coloring, and scheduling natural as CSPs rather than path-finding? Compare CSP solving (search + inference) with optimization (Chapter 2) and adversarial search (Chapter 3).\n\n**Level 1** Define CSP components: variables, domains, and constraints. Using map coloring as the running example, give pseudocode for MAKE_MAP_COLORING_CSP(regions, edges, colors). Demonstrate by hand on Australia map.\n\nThen describe backtracking search: assign variables one at a time, check consistency, backtrack on failure. Explain forward checking: after assigning variable X=v, remove inconsistent values from neighbors' domains. Give pseudocode for BACKTRACK(assignment, csp) with forward checking and domain restoration. Demonstrate by hand on a 4-coloring problem.\n\n**Level 2** Implement CSP representation and backtracking with forward checking for map coloring or Sudoku. Test with and without forward checking to see pruning effectiveness. Implement variable-ordering heuristics: MRV (minimum remaining values), degree heuristic. Implement value-ordering: least-constraining-value.\n\n**Level 3** Analyze backtracking complexity: worst-case O(d^n) where d=domain size, n=variables. Prove forward checking maintains correctness (doesn't miss solutions). Discuss why heuristics matter: MRV fails first on impossible problems, LCV succeeds first on solvable ones. Compare CSP backtracking with DFS: when is CSP formulation better? Discuss when to use CSP vs. local search (Chapter 2).\n",
      "answer": "CSP consists of variables (X₁,...,Xₙ), domains (D₁,...,Dₙ), and constraints limiting value combinations. Backtracking assigns variables sequentially, checking consistency. Forward checking prunes neighbor domains after each assignment, detecting failures early. When any domain becomes empty, backtrack. Heuristics: MRV (pick variable with fewest legal values), degree (most constraints), LCV (value constraining fewest neighbors). Complexity: O(d^n) worst-case, much faster with inference and heuristics.",
      "topics": [
        "CSP",
        "Backtracking",
        "Forward Checking",
        "Map Coloring",
        "MRV Heuristic"
      ],
      "vocab_answer": [
        {
          "word": "domain",
          "definition": "The set of possible values for a variable"
        },
        {
          "word": "constraint",
          "definition": "A restriction on which combinations of values are allowed"
        },
        {
          "word": "forward checking",
          "definition": "Pruning neighbor domains after variable assignment"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=hJ-6Ma1veUE"
      ]
    }
  },
  "content/chapter-02/questions/csp-structure.yml": {
    "type": "yaml",
    "module": {
      "id": "m4-csp-structure",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Exploit constraint graph structure to solve CSPs efficiently using tree decomposition and cutset conditioning.**\n\n**Level 0** Discuss how problem structure affects difficulty. Why are some CSPs easy despite being \"NP-hard in general\"? Explain the difference between worst-case complexity and structure-dependent complexity. What is treewidth and why does it matter? Compare with other structural properties you've seen (DAGs in search, tree game-trees).\n\n**Level 1** **Tree-Structured CSPs:** When the constraint graph is a tree (acyclic), solve in O(nd²) time. Algorithm: (1) choose arbitrary root, (2) enforce arc consistency from leaves toward root (directional consistency), (3) assign values root-to-leaves ensuring parent consistency. Give pseudocode for TREE_CSP_SOLVE(csp, root). Demonstrate by hand on a tree-shaped map.\n\n**Cutset Conditioning:** For general graphs, find a small cutset (set of variables whose removal makes graph a tree). Enumerate all assignments to cutset variables, then solve remaining tree CSP for each. Give pseudocode for CUTSET_CONDITIONING(csp, cutsetVars). Demonstrate on a nearly-tree graph with cutset size 1.\n\n**Level 2** Implement TREE_CSP_SOLVE with directional arc consistency and root-to-leaf assignment. Implement CUTSET_CONDITIONING that: (a) assigns cutset, (b) runs AC-3, (c) solves tree CSP. Compare with pure backtracking on graphs with small vs. large cutsets.\n\n**Level 3** Prove tree CSP correctness and O(nd²) complexity. Define treewidth: minimum cutset size over all tree decompositions. Prove general CSP is solvable in O(n · d^(w+1)) where w=treewidth. Discuss finding good cutsets (NP-hard in general, but heuristics work). Compare with other structured problems: cycle cutset, hypertree width. When should you use structural methods vs. general backtracking+AC-3?\n",
      "answer": "Tree CSPs solve in O(nd²): enforce directional arc consistency leaves→root, then assign root→leaves. Cutset conditioning: enumerate assignments to small variable set (cutset) that makes remaining graph a tree, solve tree CSP for each. Complexity O(d^c · nd²) where c=cutset size. Treewidth w: minimum cutset size over all decompositions. General CSPs solvable in O(n·d^(w+1)). Structure exploitation converts NP-hard to tractable.",
      "topics": [
        "Tree-Structured CSP",
        "Cutset Conditioning",
        "Treewidth",
        "Structural Decomposition"
      ],
      "vocab_answer": [
        {
          "word": "treewidth",
          "definition": "Measure of how tree-like a graph is; minimum cutset size"
        },
        {
          "word": "cutset",
          "definition": "Set of variables whose removal makes constraint graph a tree"
        },
        {
          "word": "directional consistency",
          "definition": "Arc consistency enforced in one direction through a tree"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=hJ-6Ma1veUE"
      ]
    }
  },
  "content/chapter-02/questions/horn-clause-inference.yml": {
    "type": "yaml",
    "module": {
      "id": "m5-horn-clause-inference",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Implement inference for Horn clauses using forward chaining, backward chaining, and search-based methods.**\n\n**Level 0** Discuss Horn clauses as a restricted but practical fragment of logic. Why are they computationally attractive? Compare three perspectives on inference: data-driven (forward), goal-driven (backward), and search-based (general). When is each approach preferable? Give real-world examples: expert systems, Prolog, production rules.\n\n**Level 1** Define Horn clauses: implications with at most one positive literal (B₁ ∧ B₂ ∧ ... ∧ Bₙ → H).\n\n**Forward Chaining:** Start from known facts, repeatedly apply rules whose premises are all satisfied, add conclusions to fact base. Give pseudocode for FORWARD_CHAIN(rules, facts, query). Demonstrate by hand on family relationships (parent rules → ancestor query).\n\n**Backward Chaining:** Start from query goal, find rules that conclude it, recursively prove premises. Track visited goals to avoid cycles. Give pseudocode for BACKWARD_CHAIN(rules, facts, goal, visited). Demonstrate on same example working backwards.\n\n**Inference as Search:** View inference as state-space search where states are sets of derived facts and actions are rule applications. Give pseudocode for PROOF_BFS(rules, facts, query) using BFS. Show how this generalizes both approaches.\n\n**Level 2** Implement all three methods for a knowledge base with family relationships or animal classification rules. Compare: Which derives more facts? Which is faster for single queries? Track nodes expanded in each approach.\n\n**Level 3** Prove soundness and completeness for all three methods on Horn clauses. Analyze complexity: forward chaining O(pn) where p=premises, n=facts; backward chaining can be more efficient for single queries. Discuss applications: forward in RETE algorithm (expert systems), backward in Prolog, search view connects to Chapter 1. Why doesn't forward chaining work well for queries with many irrelevant rules? When does backward chaining loop without visited tracking?\n",
      "answer": "Forward chaining (data-driven): maintain fact base, repeatedly find rules with all premises satisfied, add conclusions. Backward chaining (goal-driven): start with query, recursively prove premises of rules that conclude it, track visited to avoid cycles. Search view: states = fact sets, actions = rule applications, BFS to goal. All three are sound and complete for Horn clauses. Forward best for deriving all consequences, backward efficient for single queries, search view unifies both perspectives.",
      "topics": [
        "Forward Chaining",
        "Backward Chaining",
        "Horn Clauses",
        "Inference as Search"
      ],
      "vocab_answer": [
        {
          "word": "Horn clause",
          "definition": "A logical clause with at most one positive literal"
        },
        {
          "word": "data-driven",
          "definition": "Reasoning that starts from known facts and derives conclusions"
        },
        {
          "word": "goal-driven",
          "definition": "Reasoning that starts from a query and works backward"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=vPRDN1Y8kHg"
      ]
    }
  },
  "content/chapter-02/questions/resolution.yml": {
    "type": "yaml",
    "module": {
      "id": "m5-resolution",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Prove logical entailment using resolution and refutation.**\n\n**Level 0** Discuss the completeness of resolution: why is having a single inference rule that's both sound and complete remarkable? Compare with Horn clause methods (previous question) which are restricted to a fragment. What does \"refutation proof\" mean? Why prove by contradiction? Compare with proof strategies in mathematics.\n\n**Level 1** Define resolution rule: from clauses (A ∨ B) and (¬B ∨ C), derive (A ∨ C). Explain CNF (conjunctive normal form): conjunction of disjunctions. Describe resolution theorem proving algorithm:\n\n1. Convert KB ∪ {¬query} to CNF\n2. Repeatedly select clause pairs and resolve\n3. If empty clause ☐ derived, KB ⊨ query\n4. If no new clauses, return false\n\nGive pseudocode for RESOLUTION_ENTAILS(KB_clauses, q) and RESOLVE(c1, c2). Demonstrate by hand: prove (A→B) ∧ (B→C) ∧ A ⊨ C using resolution.\n\n**Level 2** Implement resolution theorem prover. Include CNF conversion and RESOLVE function (find complementary literals, combine remaining literals). Test on examples: modus ponens, transitivity of implication, puzzles like \"All Greeks are mortal, Socrates is Greek, therefore Socrates is mortal.\"\n\n**Level 3** Prove soundness and refutation completeness (Robinson's theorem): if KB ⊨ q, resolution will find a proof. Discuss why completeness requires refutation (can't enumerate all consequences). Analyze complexity: resolution can have exponential blowup in clause size and number. Discuss strategies: unit preference (clauses with one literal), set of support, subsumption. Compare propositional resolution with first-order resolution (requires unification). Why is resolution the basis for Prolog and modern SAT solvers?\n",
      "answer": "Resolution theorem proving works by converting KB ∪ {¬q} to CNF and repeatedly resolving pairs of clauses. If the empty clause is derived, KB ⊨ q.\n\nPseudocode:\n```\nfunction RESOLUTION_ENTAILS(KB_clauses, q):\n    clauses = set(KB_clauses ∪ CNF(¬q))\n    new = set()\n    \n    while true:\n        pairs = ALL_PAIRS(clauses)\n        for (c1,c2) in pairs:\n            resolvents = RESOLVE(c1,c2)\n            if {} in resolvents: return true      # empty clause\n            new = new ∪ resolvents\n        if new ⊆ clauses: return false\n        clauses = clauses ∪ new\n```\n",
      "topics": [
        "Resolution",
        "Propositional Logic",
        "Theorem Proving"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=7_LcT8Jw2L8"
      ]
    }
  },
  "content/chapter-02/questions/unification.yml": {
    "type": "yaml",
    "module": {
      "id": "m5-unification",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Implement unification for first-order logic terms.**\n\n**Level 0** Discuss why pattern matching is central to symbolic AI. How does unification differ from simple string matching? Why is it needed for logic programming (Prolog), theorem proving, and type inference? Compare with pattern matching in previous chapters (CSP constraints, game tree evaluation).\n\n**Level 1** Define unification: finding substitution θ such that SUBST(θ, x) = SUBST(θ, y). Explain the most general unifier (MGU). Describe the algorithm: if terms are identical, succeed; if one is a variable, bind it (after occurs check); if both are functions, must have same name/arity and unify arguments pairwise. Give pseudocode for UNIFY(x, y, subst). Demonstrate by hand: unify P(a, X, f(g(Y))) with P(Z, Z, f(W)).\n\n**Level 2** Implement UNIFY(x, y, subst) with occurs check to prevent infinite structures like X = f(X). Test on examples: Loves(X, X) with Loves(John, Y); P(X, f(X)) with P(g(Y), Z); cases that fail.\n\n**Level 3** Prove UNIFY returns the most general unifier when one exists. Analyze time complexity (near-linear with union-find optimization, but potentially exponential without occurs check). Discuss role in resolution theorem proving, Prolog query evaluation, and Hindley-Milner type inference. Why is occurs check sometimes omitted in Prolog for efficiency?\n",
      "answer": "Unification finds substitutions θ such that SUBST(θ, x) = SUBST(θ, y). Algorithm recursively processes terms: if both are identical, success; if one is a variable, extend substitution; if both are functions with same name/arity, unify arguments pairwise. Occurs check prevents X from unifying with f(X). Returns most general unifier or failure.",
      "topics": [
        "Unification",
        "First-Order Logic",
        "Pattern Matching"
      ],
      "vocab_answer": [
        {
          "word": "substitution",
          "definition": "A mapping from variables to terms"
        },
        {
          "word": "occurs check",
          "definition": "Test preventing a variable from unifying with a term containing itself"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=vXqUZNpPJ8Y"
      ]
    }
  },
  "content/chapter-03/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Probabilistic Reasoning and Learning",
          "concepts": [
            {
              "name": "Bayesian Networks, Hidden Markov Models, and Probabilistic Inference",
              "description": "Probabilistic graphical models",
              "exam_questions": [
                "questions/bayesian-networks.yml",
                "questions/temporal-reasoning.yml",
                "questions/supervised-learning.yml"
              ]
            }
          ]
        },
        {
          "category": "Neural Networks and Transformers",
          "concepts": [
            {
              "name": "Neural Networks, Backpropagation, and Transformers",
              "description": "Deep learning architectures",
              "exam_questions": [
                "questions/feedforward-backprop.yml",
                "questions/convolution.yml",
                "questions/transformer-attention.yml",
                "questions/transformer-block.yml",
                "questions/language-model-training.yml"
              ]
            }
          ]
        }
      ]
    }
  },
  "content/chapter-03/questions/bayesian-networks.yml": {
    "type": "yaml",
    "module": {
      "id": "m6-bayesian-networks",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Perform exact and approximate inference in Bayesian networks.**\n\n**Level 0** Discuss why probabilistic reasoning matters in AI. How do Bayesian networks exploit conditional independence to compactly represent joint distributions? Compare with propositional logic (Chapter 5): deterministic vs. uncertain reasoning. Why is inference hard in general (NP-hard for exact, but tractable for tree-structured networks)?\n\n**Level 1** Define Bayesian networks: DAG where nodes are random variables, edges encode dependencies, each node has a CPT (conditional probability table) P(X|Parents(X)). Explain that joint = ∏ P(Xi|Parents(Xi)).\n\n**Exact Inference (Enumeration):** To compute P(X|e), sum out hidden variables: P(X|e) = α Σ_h P(X,e,h). Give pseudocode for ENUM_ASK(X, e, bn) and ENUM_ALL(vars, e, bn). Demonstrate by hand on a simple network (e.g., Alarm → Burglary, Earthquake → Alarm).\n\n**Approximate Inference (Likelihood Weighting):** When exact inference is too expensive, sample. Generate N samples by: follow topological order, if variable is evidence fix its value and multiply weight by P(evidence|parents), otherwise sample from P(variable|parents). Give pseudocode for LIKELIHOOD_WEIGHTING(X, e, bn, N) and WEIGHTED_SAMPLE(bn, e). Demonstrate sampling process.\n\n**Level 2** Implement both methods for a Bayesian network with 4-5 variables. Test: compute P(Burglary|Alarm=true) exactly via enumeration, then approximate with 1000 weighted samples. Compare accuracy and runtime.\n\n**Level 3** Analyze exact inference complexity: naive enumeration O(d^n) where d=domain size, n=variables. Discuss variable elimination: exploit structure to compute in O(d^w) where w=treewidth (analogous to CSP tree-decomposition). Prove likelihood weighting converges to true posterior as N→∞. Discuss why it can be inefficient when evidence is unlikely. Compare with other approximate methods: rejection sampling (wastes samples), Gibbs sampling (MCMC), particle filtering (temporal). When to use each?\n",
      "answer": "Bayesian networks use DAG structure and CPTs to represent joint distributions compactly: P(X₁,...,Xₙ) = ∏P(Xᵢ|Parents(Xᵢ)). Exact inference (enumeration): sum over hidden variables, O(d^n) complexity. Variable elimination reduces to O(d^w) where w=treewidth. Approximate inference (likelihood weighting): generate weighted samples by fixing evidence and sampling non-evidence, weight by P(evidence|parents). Converges to true posterior as N→∞, but can be inefficient with unlikely evidence.",
      "topics": [
        "Bayesian Networks",
        "Probabilistic Inference",
        "Likelihood Weighting",
        "Variable Elimination"
      ],
      "vocab_answer": [
        {
          "word": "CPT",
          "definition": "Conditional Probability Table defining P(X|Parents(X))"
        },
        {
          "word": "evidence",
          "definition": "Observed variable values"
        },
        {
          "word": "likelihood weighting",
          "definition": "Sampling method that fixes evidence and weights by its probability"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=G-zirzQFWmk"
      ]
    }
  },
  "content/chapter-03/questions/convolution.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-convolution",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Implement convolutional layers for computer vision.**\n\n**Level 0** Discuss why fully-connected layers are impractical for images: an image with 224×224×3 pixels has ~150K dimensions, requiring millions of parameters. How do convolutions solve this? Three key ideas: (1) local connectivity (each neuron only sees a small patch), (2) weight sharing (same filter applied everywhere), (3) translation invariance (detecting edges anywhere in the image). Compare with biological vision: receptive fields in the visual cortex.\n\n**Level 1** Define 2D convolution: slide a k×k filter (kernel) across an H×W image, computing dot products. For position (i,j), output = Σ_a Σ_b image[i+a][j+b] * kernel[a][b]. Output dimensions: (H-k+1) × (W-k+1) with stride=1, no padding.\n\nExtensions: **Stride** s (skip s pixels per step), **Padding** p (add p zeros around border), **Multiple channels** (RGB input, multiple filters). Output size: ⌊(H+2p-k)/s⌋ + 1.\n\nGive pseudocode for CONV2D(image, kernel, stride=1, padding=0). Demonstrate 3×3 filter detecting horizontal edges on a simple image.\n\n**Level 2** Implement CONV2D with stride and padding. Test filters: vertical edge detector [[1,0,-1],[1,0,-1],[1,0,-1]], horizontal edge detector (transpose), Gaussian blur. Implement a simple CNN: conv→ReLU→maxpool→conv→ReLU→maxpool→flatten→fully-connected. Train on MNIST or CIFAR-10.\n\n**Level 3** Analyze parameter efficiency: fully-connected layer for 224×224 images with 256 hidden units = 12M params. Convolutional layer with 256 filters of size 3×3 = 2K params (1000× reduction). Discuss receptive field growth through layers. Compare CNN architectures: LeNet (1998), AlexNet (2012), VGG (deeper stacks), ResNet (skip connections), modern vision transformers (ViT replaces convolution with attention). Why did CNNs dominate vision 2012-2020? Why are transformers now competitive?\n",
      "answer": "Convolution is a fundamental operation in CNNs that applies a filter (kernel) across an image to detect features. The kernel slides over the image, computing dot products at each position.\n\nPseudocode:\n```\nfunction CONV2D(image, kernel):\n    H = image.height; W = image.width\n    kH = kernel.height; kW = kernel.width\n    outH = H - kH + 1\n    outW = W - kW + 1\n    out = matrix(outH, outW, 0)\n    \n    for i in 0..outH-1:\n        for j in 0..outW-1:\n            s = 0\n            for a in 0..kH-1:\n                for b in 0..kW-1:\n                    s += image[i+a][j+b] * kernel[a][b]\n            out[i][j] = s\n    return out\n```\n",
      "topics": [
        "Convolution",
        "CNNs",
        "Computer Vision"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=YRhxdVk_sIs"
      ]
    }
  },
  "content/chapter-03/questions/feedforward-backprop.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-feedforward-backprop",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Implement feedforward neural networks and backpropagation.**\n\n**Level 0** Discuss why neural networks are \"universal approximators\" yet were impractical until recent decades. What changed: compute (GPUs), data (internet-scale), and algorithms (better initialization, ReLU, normalization). Compare neural networks with previous AI approaches: logic (exact rules), search (discrete actions), probability (graphical models). Why are differentiable parametric models powerful?\n\n**Level 1** Define feedforward network: stack of layers f(x) = f_L(...f_2(f_1(x))). Each layer: z = Wx + b, h = activation(z). Common activations: ReLU(x) = max(0,x), sigmoid, tanh. Output layer: softmax for classification p_i = exp(z_i)/Σexp(z_j). Loss: cross-entropy L = -log(p_y).\n\nDescribe backpropagation: compute ∂L/∂W via chain rule. For 2-layer MLP:\n- Forward: z1 = W1*x + b1, h = ReLU(z1), z2 = W2*h + b2, p = softmax(z2)\n- Loss: L = -log(p[y])\n- Backward: ∂L/∂z2 = p (with p[y] -= 1), ∂L/∂W2 = ∂L/∂z2 * h^T, ∂L/∂h = W2^T * ∂L/∂z2, ∂L/∂z1 = ∂L/∂h ⊙ ReLU'(z1), ∂L/∂W1 = ∂L/∂z1 * x^T\n- Update: W -= lr * ∂L/∂W\n\nGive pseudocode for one training step: TRAIN_STEP(x, y, params, lr). Demonstrate by hand with 2 inputs, 2 hidden units, 2 outputs.\n\n**Level 2** Implement 2-layer MLP for MNIST digit classification. Include forward pass, cross-entropy loss, backward pass computing all gradients, and parameter updates. Train for multiple epochs, track training loss and test accuracy.\n\n**Level 3** Prove universal approximation theorem (sketch): a 2-layer network with enough hidden units can approximate any continuous function. Discuss why depth helps: hierarchical features, parameter efficiency. Analyze gradient flow: vanishing/exploding gradients in deep networks, solutions (ReLU, residual connections, batch norm). Compare optimizers: SGD, momentum, Adam (adaptive learning rates). Discuss initialization: Xavier/He initialization prevents vanishing activations. Why does backpropagation scale: O(parameters) time via automatic differentiation.\n",
      "answer": "Forward pass: z1 = W1*x + b1, h = ReLU(z1), z2 = W2*h + b2, p = softmax(z2). Loss: L = -log(p[y]). Backward pass: dz2 = p (with dz2[y] -= 1 for cross-entropy gradient), propagate through W2 to get dh, apply ReLU derivative to get dz1, compute weight gradients dW1, dW2, db1, db2. Update: W -= lr * dW.",
      "topics": [
        "Neural Networks",
        "Backpropagation",
        "Gradient Descent"
      ],
      "vocab_answer": [
        {
          "word": "activation function",
          "definition": "Non-linear function applied to neuron outputs (e.g., ReLU, sigmoid)"
        },
        {
          "word": "gradient",
          "definition": "Vector of partial derivatives indicating direction of steepest increase"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=Ilg3gGewQ5U"
      ]
    }
  },
  "content/chapter-03/questions/language-model-training.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-language-model-training",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Implement the training loop for autoregressive language models.**\n\n**Level 0** Discuss the deceptive simplicity of LLM training: predict the next token. This single objective yields: grammar, facts, reasoning, coding ability. Why does next-token prediction work so well? It's a dense supervision signal (every token provides feedback), and language contains compressed knowledge about the world. Compare with: image classification (single label per image), reinforcement learning (sparse rewards). Historical context: statistical language models (n-grams), neural LMs (RNNs), transformer LMs (GPT, 2018), scaling laws (2020).\n\n**Level 1** Define autoregressive language modeling: P(text) = P(t₁) P(t₂|t₁) P(t₃|t₁,t₂) ... P(tₙ|t₁,...,tₙ₋₁). At training time, given sequence [t₁, t₂, ..., tₙ], predict tᵢ₊₁ from [t₁,...,tᵢ] for all i. Use causal masking so position i cannot see future tokens.\n\n**Training pipeline:**\n1. **Tokenization:** Convert text to integers via BPE/WordPiece tokenizer (vocab ~50K)\n2. **Batching:** Sample sequences of length L (e.g., 2048) from corpus\n3. **Forward:** Model outputs logits (T × vocab_size)\n4. **Loss:** Cross-entropy between logits[i] and target[i+1], averaged over all positions\n5. **Backward:** Compute gradients via backpropagation\n6. **Update:** Apply optimizer (AdamW with lr schedule, weight decay)\n\nGive pseudocode for TRAIN_LM(corpus, tokenizer, model, optimizer, steps, seqLen, batchSize). Demonstrate loss calculation on 5-token sequence.\n\n**Level 2** Implement LM training on small corpus (e.g., TinyShakespeare). Use 6-layer transformer with 512 model dim, 8 heads, context length 256. Track perplexity = exp(avg_loss). Implement teacher forcing: use ground truth tokens as context, not model predictions. Generate text via sampling: given prompt, predict next token, append, repeat. Compare temperature: low (confident, repetitive) vs high (diverse, incoherent).\n\n**Level 3** Analyze scaling laws: loss scales as power law in compute, data, parameters (L ∝ C^(-α)). GPT-3 (175B params) vs GPT-4 (rumored 1T+): bigger is better, but diminishing returns. Discuss training challenges: (1) memory - mixed precision (FP16), gradient checkpointing, model parallelism, (2) stability - gradient clipping, warm-up schedule, (3) data - curriculum learning, data filtering. Compare training objectives: next-token (GPT), masked (BERT), denoising (T5). Discuss instruction tuning (fine-tuning on QA pairs) and RLHF (reinforcement learning from human feedback). Why GPT architecture won: simplicity, parallelization, emergent capabilities at scale.\n",
      "answer": "Training a language model involves predicting the next token given previous tokens. Sample sequences from the corpus, compute cross-entropy loss on next-token predictions, and update parameters via backpropagation.\n\nPseudocode:\n```\nfunction TRAIN_LM(corpusText, tokenizer, model, optimizer, steps, seqLen, batchSize):\n    tokens = tokenizer.encode(corpusText)\n    \n    for step in 1..steps:\n        batchX = []\n        batchY = []\n        for b in 1..batchSize:\n            i = randomInt(0, len(tokens)-seqLen-2)\n            x = tokens[i : i+seqLen]              # input tokens\n            y = tokens[i+1 : i+seqLen+1]          # next-token targets\n            batchX.append(x)\n            batchY.append(y)\n        \n        logits = model.forward(batchX)            # (B, seqLen, vocab)\n        L = CROSS_ENTROPY(logits, batchY)         # token-wise\n        grads = model.backward(L)\n        optimizer.step(model.params, grads)\n    \n    return model\n```\n",
      "topics": [
        "Language Models",
        "Transformers",
        "Next-Token Prediction"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=kCc8FmEb1nY"
      ]
    }
  },
  "content/chapter-03/questions/supervised-learning.yml": {
    "type": "yaml",
    "module": {
      "id": "m6-supervised-learning",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Implement the supervised learning pipeline: training and evaluation.**\n\n**Level 0** Discuss how supervised learning differs from the inference-based AI of previous chapters. Instead of reasoning with given knowledge (logic, probability), we learn from data. Why is this paradigm shift crucial for modern AI? Compare learning from examples vs. hand-coding rules. What is the generalization problem: performing well on unseen data?\n\n**Level 1** Define supervised learning: learn function f: X → Y from labeled examples {(x₁,y₁),...,(xₙ,yₙ)}. Describe the training pipeline:\n\n1. **Split data:** Training set (learn parameters) and test set (evaluate generalization)\n2. **Training loop:** For each epoch, iterate through training examples:\n   - Forward pass: ŷ = model(x)\n   - Loss: L = loss_fn(ŷ, y)  \n   - Backward pass: compute gradients ∇L\n   - Update: params ← params - α∇L (gradient descent)\n3. **Evaluation:** Compute accuracy/error on test set\n\nGive pseudocode for TRAIN_EVAL(dataset, model, lossFn, optimizer, epochs). Demonstrate gradient descent by hand on simple linear regression.\n\n**Level 2** Implement the pipeline for classification (e.g., digit recognition or iris species). Use a simple neural network with one hidden layer. Include: data loading/splitting, forward/backward passes, SGD optimizer, training loop with loss tracking, test evaluation. Visualize training loss over epochs. Compare training vs. test accuracy to detect overfitting.\n\n**Level 3** Analyze generalization theory: training error vs. test error, overfitting (model too complex), underfitting (too simple). Discuss regularization techniques: L2 penalty, dropout, early stopping. Prove that gradient descent converges for convex loss functions (linear regression). For non-convex (neural networks), discuss local minima and SGD noise benefits. Compare optimization algorithms: SGD, momentum, Adam. Discuss hyperparameter tuning: learning rate, batch size, architecture. How does this connect to other AI: Bayesian learning (prior over models), reinforcement learning (learning from rewards instead of labels)?\n",
      "answer": "Supervised learning trains a model on labeled data. Split data into training and test sets, iterate through epochs updating parameters via gradient descent, then evaluate accuracy on test set.\n\nPseudocode:\n```\nfunction TRAIN_EVAL(dataset, model, lossFn, optimizer, epochs):\n    (trainSet, testSet) = SPLIT(dataset, ratio=0.8, shuffle=true)\n    \n    for e in 1..epochs:\n        for (x,y) in trainSet:\n            yhat = model.forward(x)\n            L = lossFn(yhat, y)\n            grads = model.backward(L)\n            optimizer.step(model.params, grads)\n    \n    correct = 0\n    total = 0\n    for (x,y) in testSet:\n        yhat = argmax(model.forward(x))\n        if yhat == y: correct += 1\n        total += 1\n    return correct / total\n```\n",
      "topics": [
        "Supervised Learning",
        "Machine Learning",
        "Training Pipeline"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=aircAruvnKk"
      ]
    }
  },
  "content/chapter-03/questions/temporal-reasoning.yml": {
    "type": "yaml",
    "module": {
      "id": "m6-temporal-reasoning",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Perform exact and approximate filtering for temporal probabilistic models.**\n\n**Level 0** Discuss why temporal reasoning matters: robotics, speech recognition, autonomous systems. How do Hidden Markov Models extend Bayesian networks to time? Compare filtering (estimating current state) with smoothing (past states) and prediction (future states). Why is the Markov assumption (state at t depends only on t-1) both limiting and enabling?\n\n**Level 1** Define HMM components: state space X, transition model P(Xt|Xt-1), observation model P(Et|Xt), initial belief P(X₀).\n\n**Exact Filtering (Forward Algorithm):** Maintain belief distribution b_t = P(Xt|e1:t). Recursive update: (1) Predict: b̄_{t+1} = Σ P(Xt+1|xt) b_t(xt), (2) Update: b_{t+1} = α P(et+1|Xt+1) b̄_{t+1}. Give pseudocode for HMM_FILTER(priorBelief, observations, T, O). In matrix form: b_{t+1} = α O_{t+1} T^T b_t. Demonstrate by hand: robot localization with 3 positions.\n\n**Approximate Filtering (Particle Filter):** When state space is large or continuous, represent belief with N weighted particles. Algorithm: (1) Propagate: sample each particle through transition model, (2) Weight: multiply by observation likelihood P(e|x), (3) Resample: draw N new particles proportional to weights. Give pseudocode for PARTICLE_FILTER(particles, observation, TRANSITION, OBS_LIK, N). Demonstrate with continuous state space.\n\n**Level 2** Implement both methods. **HMM filtering:** use transition and observation matrices, update belief vector at each time step. **Particle filtering:** implement propagate-weight-resample loop for 2D robot localization with noisy odometry and range sensors. Compare: exact filter with 100 discrete states vs. 1000 particles.\n\n**Level 3** Analyze complexity: HMM forward algorithm O(n²T) where n=states, T=time steps. Particle filtering O(NT) but handles continuous/large state spaces. Discuss particle degeneracy (all weight on few particles) and solutions: resampling strategies, importance sampling. Compare with other temporal inference: Viterbi (most likely sequence), Kalman filter (Gaussian beliefs, linear dynamics), extended/unscented Kalman filters. Applications: SLAM (simultaneous localization and mapping), speech recognition (phoneme sequences), activity recognition.\n",
      "answer": "HMM filtering maintains belief state P(Xt|e1:t) via predict-update: predict b̄t+1 = T^T bt, update bt+1 = α Ot+1 b̄t+1. O(n²) per step. Particle filtering approximates with N samples: propagate through transition, weight by P(e|x), resample proportional to weights. Handles continuous/large state spaces. O(N) per step but particle degeneracy can be problematic. Both are Bayesian updates combining prior dynamics with new observations.",
      "topics": [
        "Hidden Markov Models",
        "Particle Filtering",
        "Temporal Reasoning",
        "Bayesian Filtering"
      ],
      "vocab_answer": [
        {
          "word": "filtering",
          "definition": "Estimating current state from past observations"
        },
        {
          "word": "belief state",
          "definition": "Probability distribution over possible hidden states"
        },
        {
          "word": "particle",
          "definition": "A sample representing one possible state hypothesis"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=kqSzLo9fenk"
      ]
    }
  },
  "content/chapter-03/questions/transformer-attention.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-transformer-attention",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Implement self-attention, the core mechanism of transformers.**\n\n**Level 0** Discuss why RNNs were limiting: sequential processing prevents parallelization, long sequences suffer from vanishing gradients. How does attention solve this? Direct connections between all positions, allowing information flow in O(1) steps (not O(T) sequential steps). Why is this revolutionary? Transformers can process entire sequences in parallel during training, scaling to millions of tokens. Compare with previous sequence models: HMMs (Chapter 6) use fixed Markov assumptions, RNNs maintain hidden state sequentially.\n\n**Level 1** Define self-attention: given input sequence X (T × d), compute Q=XW_Q, K=XW_K, V=XW_V (learned projections). Attention scores: S = QK^T/√d (T × T matrix where S[i,j] = how much position i attends to position j). Weights: A = softmax(S, axis=1) (each row sums to 1). Output: Y = AV (weighted sum of values).\n\nWhy scaling by √d? Prevents dot products from growing large (softmax saturates). Why QKV separate? Q = \"what I'm looking for\", K = \"what I have\", V = \"content to retrieve\".\n\nGive pseudocode for ATTENTION(Q, K, V). Demonstrate by hand on 3-token sequence.\n\n**Level 2** Implement single-head self-attention. Test on simple sequences: (1) positional patterns (attending to previous tokens), (2) content-based (attending to similar words). Visualize attention weights as heatmap. Implement multi-head attention: run h parallel attention heads with different W_Q, W_K, W_V, concatenate outputs, project with W_O.\n\n**Level 3** Analyze complexity: attention is O(T²d) for sequence length T. Why is this problematic for long sequences? Discuss efficient variants: sparse attention, linear attention, Flash Attention (memory-efficient). Explain multi-head benefits: different heads learn different patterns (syntax, semantics, positional). Compare attention with: (1) RNN: O(T) sequential steps vs O(1) parallel, (2) CNN: fixed receptive field vs global context. Discuss causal masking for autoregressive LMs: prevent position i from attending to j>i. Why did attention succeed? Parallelization + long-range dependencies + no inductive bias (learns patterns from data).\n",
      "answer": "Self-attention computes attention(Q,K,V) = softmax(QK^T/sqrt(d)) * V, where Q,K,V are learned linear projections of input. Each position attends to all positions (unlike RNNs' sequential processing). Transformer blocks stack attention + feedforward with residual connections and layer norm. This parallel architecture enables efficient training on GPUs and captures long-range dependencies better than RNNs.",
      "topics": [
        "Attention Mechanisms",
        "Transformers",
        "Self-Attention"
      ],
      "vocab_answer": [
        {
          "word": "query",
          "definition": "Vector representing what information a position is looking for"
        },
        {
          "word": "key",
          "definition": "Vector representing what information a position contains"
        },
        {
          "word": "value",
          "definition": "Vector containing the actual information to be retrieved"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=4Bdc55j80l8"
      ]
    }
  },
  "content/chapter-03/questions/transformer-block.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-transformer-block",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Implement a complete transformer block.**\n\n**Level 0** Discuss the transformer architecture philosophy: repeating identical blocks stacked deeply (GPT-3 has 96 blocks). Why is modularity powerful? Each block refines representations, and stacking allows learning hierarchical patterns. Compare with CNN (convolutional blocks) and ResNet (residual blocks). What innovations made deep transformers trainable: residual connections (gradient flow), layer normalization (stable activations), careful initialization?\n\n**Level 1** Define transformer block components:\n\n1. **Multi-head self-attention sublayer:**\n   - Compute attention: A = MultiHeadAttention(X)\n   - Residual connection: X' = X + A  \n   - Layer norm: X1 = LayerNorm(X')\n\n2. **Position-wise feedforward sublayer:**\n   - Two-layer MLP: F = GELU(X1*W1 + b1)*W2 + b2\n   - Residual: X'' = X1 + F\n   - Layer norm: Y = LayerNorm(X'')\n\n**Residual connections:** X + f(X) allows gradients to flow directly backward. **Layer normalization:** normalize across features (not batch), stabilizes training.\n\nGive pseudocode for TRANSFORMER_BLOCK(X, params). Show dimensions at each step for T=10 tokens, d=512 model dimension.\n\n**Level 2** Implement transformer block with multi-head attention (8 heads), feedforward (4d intermediate dimension), GELU activation, residuals, and layer norm. Stack 6 blocks to build encoder. Test on sequence classification task. Visualize attention patterns in different layers: early layers learn positional patterns, later layers learn semantic relations.\n\n**Level 3** Analyze residual connection theory: without residuals, gradients ∝ (W^T)^L vanish/explode. With residuals: gradient highway, effective depth is adaptive. Discuss layer norm vs batch norm: layer norm normalizes per example (independent of batch), crucial for variable-length sequences. Compare transformer variants: encoder-only (BERT, classification), decoder-only (GPT, generation), encoder-decoder (T5, translation). Discuss architectural choices: pre-norm vs post-norm (where layer norm is placed), GLU vs GELU activation, learned vs sinusoidal positional encoding. Why 512-768-1024 model dimensions? Power of 2 for GPU efficiency, large enough for rich representations.\n",
      "answer": "A transformer block combines multi-head self-attention with a feedforward network, using residual connections and layer normalization. This architecture enables powerful sequence modeling.\n\nPseudocode:\n```\nfunction TRANSFORMER_BLOCK(X, params):\n    # X: (seqLen x dModel)\n    A = ATTENTION(X*params.Wq, X*params.Wk, X*params.Wv) * params.Wo\n    X1 = LAYERNORM(X + A)\n    \n    F = GELU(X1*params.W1 + params.b1) * params.W2 + params.b2\n    Y = LAYERNORM(X1 + F)\n    return Y\n```\n",
      "topics": [
        "Transformers",
        "Deep Learning",
        "Attention Mechanisms"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=4Bdc55j80l8"
      ]
    }
  },
  "index.md": {
    "type": "markdown",
    "module": "# CS440: Artificial Intelligence\r\n\r\nWelcome to CS440! This course explores the foundational concepts and techniques of artificial intelligence, from intelligent agents and search algorithms to machine learning and beyond.\r\n\r\n## Course Overview\r\n\r\nThis course provides a comprehensive introduction to artificial intelligence, covering fundamental concepts, algorithms, and paradigms that enable machines to exhibit intelligent behavior. Through hands-on problems and theoretical exploration, students will master core AI techniques including search algorithms, knowledge representation, planning, reasoning under uncertainty, and machine learning fundamentals.\r\n\r\n## Topics\r\n\r\n### [Module 1: Search and Games](content/chapter-01/)\r\n\r\n**Intelligent Agents and Search:**\r\n* State spaces and problem representation (8-puzzle)\r\n* Uninformed search: BFS, Iterative Deepening DFS\r\n* Informed search: A* with admissible heuristics\r\n* Maze navigation algorithms\r\n\r\n**Local Search and Optimization:**\r\n* Hill climbing and local optima (8-Queens problem)\r\n* Stochastic methods: Simulated annealing, Genetic algorithms\r\n* Continuous optimization\r\n\r\n**Adversarial Search:**\r\n* Minimax algorithm and alpha-beta pruning\r\n* Expectiminimax for stochastic games\r\n* Monte Carlo tree search (MCTS)\r\n* Nash equilibrium basics\r\n\r\n---\r\n\r\n### [Module 2: Reasoning with Constraints and Logic](content/chapter-02/)\r\n\r\n**Constraint Satisfaction Problems:**\r\n* CSP formulation: Variables, domains, constraints\r\n* Backtracking with forward checking\r\n* Arc consistency (AC-3 algorithm)\r\n* Tree-structured CSPs and cutset conditioning\r\n\r\n**Logic and Inference:**\r\n* Unification algorithm and term matching\r\n* Forward and backward chaining for Horn clauses\r\n* Resolution theorem proving\r\n\r\n---\r\n\r\n### [Module 3: Probabilistic Reasoning and Deep Learning](content/chapter-03/)\r\n\r\n**Probabilistic Reasoning:**\r\n* Bayesian Networks: Exact inference and likelihood weighting\r\n* Hidden Markov Models (HMMs) and filtering\r\n* Particle filtering for temporal reasoning\r\n* Supervised learning fundamentals\r\n\r\n**Neural Networks and Transformers:**\r\n* Feedforward networks and backpropagation\r\n* Convolutional networks for computer vision\r\n* Self-attention mechanism\r\n* Transformer blocks and language model training\r\n\r\n---\r\n\r\n## Getting Started\r\n\r\nNavigate to the module content using the navigation panel on the left. Each module contains:\r\n\r\n- **Concept explanations** with multiple complexity levels\r\n- **Practice questions** to test your understanding\r\n- **Video resources** for visual learners\r\n- **Hands-on coding exercises** to implement AI algorithms\r\n\r\nGood luck with your AI journey!\r\n"
  }
};

export const stats = {
  yamlCount: 29,
  markdownCount: 1,
  totalCount: 30
};
