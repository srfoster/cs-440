// Auto-generated compiled content - DO NOT EDIT
// Generated on 2026-01-12T22:25:44.254Z

export const compiledFiles = {
  "content/chapter-01/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Intelligent Agents and Search",
          "concepts": [
            {
              "name": "Problem Formulation",
              "description": "Defining states, actions, and goals for search problems",
              "exam_questions": [
                "questions/8puzzle-formulation.yml"
              ]
            },
            {
              "name": "Uninformed Search",
              "description": "Search strategies that don't use problem-specific knowledge",
              "exam_questions": [
                "questions/maze-navigation.yml",
                "questions/bfs-iddfs.yml",
                "questions/depth-cutoff.yml"
              ]
            },
            {
              "name": "Informed Search",
              "description": "Search algorithms that use heuristics to guide exploration",
              "exam_questions": [
                "questions/astar-heuristic.yml"
              ]
            },
            {
              "name": "Multi-Agent Systems",
              "description": "Simulating multiple interacting agents",
              "exam_questions": [
                "questions/boids.yml"
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
      "question": "**Write the problem formulation functions for the 8-puzzle problem.**\n\n**Level 0** Discuss the big-picture significance of problem formulation in AI. Why is it important to formally define states, actions, and goals?\n\n**Level 1** Explain the 8-puzzle problem intuitively. Describe how to represent the state as a 3×3 grid, what actions are available, and how to test if we've reached the goal. Give pseudocode for ACTIONS(s), RESULT(s,a), and GOAL_TEST(s).\n\n**Level 2** Implement the three functions in a language of your choice. Your implementation should handle a state represented as a 3×3 grid where 0 represents the blank tile.\n\n**Level 3** Analyze the state space size for the 8-puzzle. Discuss reachability (not all 9!/2 states are reachable from a given start) and the complexity implications for search algorithms.\n",
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
  "content/chapter-01/questions/astar-heuristic.yml": {
    "type": "yaml",
    "module": {
      "id": "m1-astar-heuristic",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Implement A* search with an admissible heuristic.**\n\n**Level 0** Discuss the importance of informed search in AI. How do heuristics help guide search algorithms toward goals more efficiently?\n\n**Level 1** Explain A* search intuitively: how it combines actual cost g(n) and estimated future cost h(n) to prioritize which states to explore. Provide pseudocode for A* including the use of a priority queue and the f = g + h evaluation function.\n\n**Level 2** Implement A* search for a maze navigation problem. Include a heuristic function (such as Manhattan distance) and proper handling of the open list (priority queue) and closed list.\n\n**Level 3** Prove that A* is optimal when the heuristic is admissible. Analyze the time and space complexity. Discuss the impact of heuristic quality on performance.\n",
      "answer": "A* search uses f(n) = g(n) + h(n) where g(n) is the actual cost from start to n and h(n) is the estimated cost from n to goal. It maintains a priority queue (open list) ordered by f-value. A* is optimal when h is admissible (never overestimates). Better heuristics (higher h while staying admissible) explore fewer nodes. Manhattan distance is a classic admissible heuristic for grid-based problems.",
      "answer_kindergarten": "A* is like having a smart helper when you're trying to find your way somewhere. Instead of just trying every path randomly, your helper guesses which paths look like they might be shorter. The helper keeps track of how far you've already walked, plus guesses how much further you need to go, and always picks the path that seems best overall.",
      "answer_3rd_grade": "A* search is smarter than regular search because it uses clues to find the goal faster. It keeps two numbers for each position: how many steps you've taken to get there (g), and your guess about how many more steps you need (h). It adds these together (f = g + h) and always explores the position with the smallest total first. For a maze, you can guess the remaining distance by counting squares horizontally and vertically (Manhattan distance).",
      "answer_7th_grade": "A* maintains a priority queue of states ordered by f(n) = g(n) + h(n), where g(n) is the actual cost from the start and h(n) is the heuristic estimate to the goal. It repeatedly extracts the state with minimum f-value, checks if it's the goal, and generates its successors. For each successor, if it offers a better path (lower g-value) than previously found, A* updates its g-value and parent, and adds it to the priority queue with its f-value as priority. For grid mazes, Manhattan distance |x1-x2| + |y1-y2| is an admissible heuristic.",
      "answer_high_school": "A* is an informed search algorithm that expands states in order of increasing f(n) = g(n) + h(n). The algorithm maintains: (1) an open list (min-priority queue by f-value), (2) g-scores (distance from start), and (3) parent pointers for path reconstruction. The heuristic h(n) must be admissible (h(n) ≤ h*(n) where h* is true cost) to guarantee optimality. A* is optimal because it never expands a state with f(n) > C* (optimal cost) before expanding all states on an optimal path. Consistency (h(n) ≤ c(n,n') + h(n')) implies admissibility and ensures states are expanded at most once.",
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
  "content/chapter-01/questions/bfs-iddfs.yml": {
    "type": "yaml",
    "module": {
      "id": "m1-bfs-iddfs",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Implement BFS (Breadth-First Search) and IDDFS (Iterative Deepening Depth-First Search) algorithms.**\n\n**Level 0** Discuss why we need different search strategies. What are the trade-offs between memory usage and completeness?\n\n**Level 1** Explain how BFS explores states level by level using a queue, and how IDDFS combines the benefits of BFS and DFS by repeatedly performing depth-limited searches with increasing depth limits. Provide pseudocode for both algorithms.\n\n**Level 2** Implement BFS and IDDFS for a generic state space using ACTIONS, RESULT, and GOAL_TEST functions. Include path reconstruction.\n\n**Level 3** Analyze the time and space complexity of both algorithms. Compare their performance characteristics: when is IDDFS preferred over BFS despite exploring some states multiple times?\n",
      "answer": "BFS uses a queue (FIFO) to explore states level by level, guaranteeing the shortest path but requiring O(b^d) space. IDDFS performs repeated depth-limited searches with increasing depth limits, achieving BFS's completeness and optimality with only O(bd) space. IDDFS is preferred when memory is limited and the branching factor is high, even though it revisits states.",
      "answer_kindergarten": "Imagine finding your way through a maze. BFS is like bringing lots of friends to explore every path at the same time, level by level - you'll find the shortest way out but need lots of friends to remember everything. IDDFS is like going in yourself, trying paths that go just 1 step deep, then 2 steps, then 3 steps, and so on. You might check some paths twice, but you don't need to remember as much!",
      "answer_3rd_grade": "BFS explores all possible moves one step away, then two steps away, then three steps, and so on, until it finds the goal. It uses a line (queue) to keep track of what to explore next. It finds the shortest path but needs lots of memory. IDDFS is smarter with memory - it searches up to depth 1, then restarts and searches up to depth 2, then depth 3, and keeps going until it finds the answer.",
      "answer_7th_grade": "BFS maintains a queue of states to explore. It adds the start state, then repeatedly takes the front state from the queue, generates all its children using ACTIONS and RESULT, and adds unvisited children to the back of the queue. When it finds the goal, it reconstructs the path using parent pointers. IDDFS runs depth-limited search repeatedly with limits 0, 1, 2, 3... until finding the goal. Each depth-limited search uses recursion and a limit parameter, returning failure if the limit is reached without finding the goal.",
      "answer_high_school": "BFS is a complete, optimal graph search algorithm for uniform-cost problems. Implementation uses a FIFO queue for the frontier, a visited set to prevent cycles, and a parent map for path reconstruction. When a goal is found, backtracking through parent pointers yields the solution path. IDDFS iteratively performs depth-limited search (DLS) with increasing depth limits. DLS uses recursion with a depth parameter, returning failure if depth reaches 0 before finding a goal. IDDFS combines DFS's space efficiency O(bd) with BFS's optimality for uniform costs.",
      "answer_undergraduate": "BFS: Complete and optimal for unit-cost edges. Time O(b^d), space O(b^d) where b is branching factor and d is solution depth. Space complexity is the limiting factor for large problems. IDDFS: Iteratively executes DLS with depth cutoffs 0,1,2,...,d. Time complexity O(b^d) - appears wasteful with repeated state generation, but for trees with branching factor b>1, overhead is actually O(b^(d-1) + b^(d-2) + ...) = O(b^d / (b-1)), asymptotically similar to BFS. Space complexity O(bd) is dramatically better, making IDDFS preferred for large state spaces with reasonable branching factors. Key insight: in exponential growth, the last level dominates, so revisiting earlier levels is acceptable. IDDFS is optimal for uniform-cost problems and complete for infinite state spaces where BFS would exhaust memory.",
      "topics": [
        "Search Algorithms",
        "BFS",
        "IDDFS",
        "Completeness and Optimality"
      ],
      "vocab_answer": [
        {
          "word": "queue",
          "definition": "A first-in-first-out (FIFO) data structure"
        },
        {
          "word": "frontier",
          "definition": "The set of states that have been generated but not yet explored"
        },
        {
          "word": "depth-limited",
          "definition": "A search that stops exploring paths that exceed a specified depth"
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
  "content/chapter-01/questions/boids.yml": {
    "type": "yaml",
    "module": {
      "id": "m1-boids",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Implement one simulation tick for boids with separation, alignment, cohesion.**\n\nWrite STEP_BOIDS(boids, params) that updates all boid positions and velocities.\n",
      "answer": "Boids algorithm simulates flocking behavior using three rules: separation (avoid crowding), alignment (steer toward average heading), and cohesion (steer toward average position). Each boid computes forces from neighbors within a radius, combines them with weights, and updates velocity and position.\n\nPseudocode:\n```\nfunction STEP_BOIDS(boids, params):\n    newBoids = []\n    for i in 0..len(boids)-1:\n        b = boids[i]\n        neighbors = NEIGHBORS(boids, i, params.radius)\n        \n        sep = vector(0,0)\n        ali = vector(0,0)\n        coh = vector(0,0)\n        \n        for j in neighbors:\n            diff = b.pos - boids[j].pos\n            sep += normalize(diff) / (epsilon + norm(diff))\n            ali += boids[j].vel\n            coh += boids[j].pos\n        \n        if len(neighbors) > 0:\n            ali = (ali / len(neighbors)) - b.vel\n            coh = (coh / len(neighbors)) - b.pos\n        \n        accel = params.wSep*sep + params.wAli*ali + params.wCoh*coh\n        v2 = limit(b.vel + accel, params.maxSpeed)\n        p2 = b.pos + v2\n        newBoids.append( (p2, v2) )\n    return newBoids\n```\n",
      "topics": [
        "Multi-Agent Systems",
        "Boids",
        "Flocking Behavior"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=mhjuuHl6qHM"
      ]
    }
  },
  "content/chapter-01/questions/depth-cutoff.yml": {
    "type": "yaml",
    "module": {
      "id": "m1-depth-cutoff",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Implement depth-limited DFS (DFS with cutoff).**\n\nWrite DFS_CUTOFF(start, cutoff) that returns failure if not found within depth.\n",
      "answer": "DFS_CUTOFF performs depth-limited search by calling the DLS (depth-limited search) helper function with a specified cutoff depth. If the goal is not found within that depth, it returns failure.\n\nPseudocode:\n```\nfunction DFS_CUTOFF(start, cutoff):\n    parent = map()\n    parent[start] = null\n    found = DLS(start, cutoff, parent)\n    if found == failure: return failure\n    return RECONSTRUCT_PATH(parent, found)\n```\n",
      "topics": [
        "Depth-Limited Search",
        "DFS"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=HZ5YTanv5QE"
      ]
    }
  },
  "content/chapter-01/questions/maze-navigation.yml": {
    "type": "yaml",
    "module": {
      "id": "m1-maze-navigation",
      "type": "short-answer",
      "chapter": 1,
      "question": "**Write an algorithm to navigate a maze from a start position to a goal position.**\n\n**Level 0** Discuss the big-picture significance of maze navigation in the context of artificial intelligence and real-world problem solving.\n\n**Level 1** Discuss the intuition behind maze navigation by describing the problem representation (how to represent the maze and positions), relevant data structures, and give pseudocode for one or more search algorithms that can solve this problem. Step through your algorithm with a simple example.\n\n**Level 2** Implement your algorithm(s) in a language of your choice. Your implementation should handle a maze represented as a 2D grid where walls are obstacles and the agent must find a path from start to goal.\n\n**Level 3** Analyze the space and time complexity of your algorithm(s). Compare different search strategies (BFS, DFS, A*) for maze navigation. Discuss when each approach is most appropriate and re-contextualize maze navigation in terms of broader AI search problems.\n",
      "answer": "Maze navigation can be solved using graph search algorithms. The maze is represented as a graph where each cell is a node and edges connect adjacent passable cells. Breadth-First Search (BFS) guarantees finding the shortest path by exploring all positions at distance k before exploring positions at distance k+1. The algorithm uses a queue to track positions to explore and a visited set to avoid cycles.",
      "answer_kindergarten": "To get through a maze, you can try going in different directions from where you start. If you hit a wall, you go back and try a different way. You keep trying until you find the exit. It's like when you're in a corn maze and you remember where you've already been so you don't get lost in circles.",
      "answer_3rd_grade": "Maze navigation is like being a robot that needs to find its way from one place to another without bumping into walls. The robot remembers where it's been and tries different paths. One good way is to check all nearby spots first, then check spots a bit further away, until you find the goal. This is called Breadth-First Search.",
      "answer_7th_grade": "To navigate a maze, we represent it as a grid where each cell can be either a wall or a passable space. Using Breadth-First Search, we start at the beginning position and explore all adjacent cells (up, down, left, right). We keep track of visited cells and use a queue to process cells in the order we discover them. This guarantees we'll find the shortest path if one exists.",
      "answer_high_school": "Maze navigation is a classic graph search problem. The maze is modeled as a graph G=(V,E) where vertices are passable cells and edges connect adjacent cells. BFS explores the graph level by level using a FIFO queue, guaranteeing optimal path length. DFS uses a stack (or recursion) and may find a path faster but doesn't guarantee optimality. A* search improves on BFS by using heuristics (like Manhattan distance) to guide the search toward the goal more efficiently.",
      "answer_undergraduate": "Maze navigation exemplifies state-space search in AI. The problem can be formalized as a tuple (S, s₀, A, T, G) where S is the state space (all cells), s₀ is the start state, A are actions (move up/down/left/right), T is the transition function, and G is the goal set. BFS achieves O(|V|+|E|) time complexity and guarantees completeness and optimality for uniform cost. A* search with an admissible heuristic h(n) (h(n) ≤ h*(n)) maintains optimality while potentially reducing the search space significantly. The choice of algorithm depends on the problem characteristics: BFS for guaranteed shortest path, DFS for memory-constrained scenarios, and A* for large state spaces with good heuristics.",
      "topics": [
        "Search Algorithms",
        "Graph Traversal",
        "Problem Solving",
        "State Space"
      ],
      "vocab_answer": [
        {
          "word": "Breadth-First Search",
          "definition": "A search algorithm that explores all nodes at the current depth before moving to nodes at the next depth level"
        },
        {
          "word": "state space",
          "definition": "The set of all possible configurations or positions in a problem"
        },
        {
          "word": "heuristic",
          "definition": "A rule of thumb or educated guess that guides search toward promising solutions"
        }
      ],
      "vocab_kindergarten": [
        {
          "word": "maze",
          "definition": "A puzzle with walls and paths where you need to find your way from start to finish"
        },
        {
          "word": "path",
          "definition": "The way you go from one place to another"
        }
      ],
      "vocab_3rd_grade": [
        {
          "word": "grid",
          "definition": "A pattern of squares arranged in rows and columns, like a checkerboard"
        },
        {
          "word": "adjacent",
          "definition": "Next to each other, touching on one side"
        }
      ],
      "vocab_7th_grade": [
        {
          "word": "queue",
          "definition": "A data structure that processes items in first-in-first-out (FIFO) order, like a line at a store"
        },
        {
          "word": "visited",
          "definition": "Cells or positions that have already been explored by the search algorithm"
        }
      ],
      "vocab_high_school": [
        {
          "word": "graph",
          "definition": "A mathematical structure consisting of vertices (nodes) connected by edges"
        },
        {
          "word": "optimal",
          "definition": "The best possible solution according to some measure, like shortest path length"
        },
        {
          "word": "Manhattan distance",
          "definition": "The sum of absolute differences in coordinates, representing grid distance"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=KiCBXu4P-2Y",
        "https://www.youtube.com/watch?v=oDqjPvD54Ss",
        "https://www.youtube.com/watch?v=HZ5YTanv5QE"
      ],
      "vocab_undergraduate": [
        {
          "word": "admissible heuristic",
          "definition": "A heuristic function that never overestimates the cost to reach the goal"
        },
        {
          "word": "completeness",
          "definition": "A property of search algorithms guaranteeing that a solution will be found if one exists"
        },
        {
          "word": "time complexity",
          "definition": "A measure of how the number of operations grows as a function of input size"
        }
      ]
    }
  },
  "content/chapter-02/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Local Search and Optimization",
          "concepts": [
            {
              "name": "Objective Functions",
              "description": "Defining optimization criteria",
              "exam_questions": [
                "questions/8queens-objective.yml"
              ]
            },
            {
              "name": "Hill Climbing",
              "description": "Greedy local search that moves to better neighbors",
              "exam_questions": [
                "questions/8queens-hillclimb.yml"
              ]
            },
            {
              "name": "Simulated Annealing",
              "description": "Stochastic search with probabilistic acceptance of worse states",
              "exam_questions": [
                "questions/simulated-annealing.yml"
              ]
            },
            {
              "name": "Genetic Algorithms",
              "description": "Population-based search inspired by evolution",
              "exam_questions": [
                "questions/genetic-algorithm.yml"
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
        }
      ]
    }
  },
  "content/chapter-02/questions/8queens-hillclimb.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-8queens-hillclimb",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Implement hill climbing to solve the 8-Queens problem.**\n\n**Level 0** Discuss why local search is appropriate for problems like N-Queens. What are the advantages over systematic search?\n\n**Level 1** Explain the 8-Queens problem and how to represent a state as board[col]=row. Describe the CONFLICTS objective function and the hill climbing algorithm that iteratively moves to the best neighbor. Provide pseudocode.\n\n**Level 2** Implement hill climbing for 8-Queens including the CONFLICTS function and neighbor generation. Handle the local optima problem.\n\n**Level 3** Analyze the completeness and optimality of hill climbing. Discuss why it gets stuck in local optima and strategies to escape them (random restarts, sideways moves, etc.).\n",
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
  "content/chapter-02/questions/8queens-objective.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-8queens-objective",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Write the objective function CONFLICTS(board) for 8-Queens where board[col]=row.**\n\nCount the number of queen pairs attacking each other.\n",
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
  "content/chapter-02/questions/airport-optimization.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-airport-optimization",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Minimize sum of squared distances from candidate airport (x,y) to towns using continuous local search.**\n\nImplement AIRPORT_OBJECTIVE and RANDOM_SEARCH_CONTINUOUS.\n",
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
  "content/chapter-02/questions/genetic-algorithm.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-genetic-algorithm",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Implement a genetic algorithm for the 8-Queens problem.**\n\n**Level 0** Discuss how evolution inspires computational search. Why are population-based methods powerful?\n\n**Level 1** Explain genetic algorithm components: population, fitness, selection, crossover, and mutation. Provide pseudocode for the GA loop.\n\n**Level 2** Implement a GA with tournament selection, one-point crossover, and mutation for 8-Queens.\n\n**Level 3** Analyze GA performance: diversity vs. convergence, parameter tuning, and schema theorem basics.\n",
      "answer": "Genetic algorithms maintain a population of candidate solutions, select parents based on fitness, create offspring through crossover and mutation, and iterate for multiple generations. For 8-Queens: fitness = 1/(1+conflicts), tournament selection picks best of k random individuals, one-point crossover combines parent board segments, mutation randomly changes queen positions.",
      "topics": [
        "Genetic Algorithms",
        "Evolutionary Computation",
        "Population-based Search"
      ],
      "vocab_answer": [
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
        "https://www.youtube.com/watch?v=1i8muvzZkPw"
      ]
    }
  },
  "content/chapter-02/questions/simulated-annealing.yml": {
    "type": "yaml",
    "module": {
      "id": "m2-simulated-annealing",
      "type": "short-answer",
      "chapter": 2,
      "question": "**Implement simulated annealing for the 8-Queens problem.**\n\n**Level 0** Explain why randomness and accepting worse moves can help optimization. What real-world annealing process inspired this algorithm?\n\n**Level 1** Describe simulated annealing intuitively: how temperature controls the probability of accepting worse moves, and how cooling schedules work. Provide pseudocode.\n\n**Level 2** Implement simulated annealing with temperature schedule and probabilistic acceptance based on exp(-delta/T).\n\n**Level 3** Analyze convergence properties. Discuss the trade-off between exploration (high temperature) and exploitation (low temperature).\n",
      "answer": "Simulated annealing accepts worse moves with probability exp(-delta/T) where delta is the increase in cost and T is temperature. Temperature decreases over time (T = alpha * T). This allows escaping local optima early while converging to good solutions as T approaches 0. Named after metallurgical annealing where controlled cooling produces strong crystal structures.",
      "topics": [
        "Simulated Annealing",
        "Stochastic Search",
        "Local Search"
      ],
      "vocab_answer": [
        {
          "word": "temperature",
          "definition": "A parameter controlling randomness in simulated annealing"
        },
        {
          "word": "cooling schedule",
          "definition": "The rate at which temperature decreases over time"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=eBmU1ONJ-os"
      ]
    }
  },
  "content/chapter-03/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Adversarial Search and Games",
          "concepts": [
            {
              "name": "Minimax and Alpha-Beta",
              "description": "Optimal play in two-player zero-sum games",
              "exam_questions": [
                "questions/minimax-alphabeta.yml"
              ]
            },
            {
              "name": "Stochastic Games",
              "description": "Games with chance elements",
              "exam_questions": [
                "questions/expectiminimax.yml"
              ]
            },
            {
              "name": "Monte Carlo Methods",
              "description": "Stochastic game playing",
              "exam_questions": [
                "questions/monte-carlo.yml"
              ]
            },
            {
              "name": "Game Theory",
              "description": "Strategic reasoning and equilibria",
              "exam_questions": [
                "questions/nash-equilibrium.yml"
              ]
            }
          ]
        }
      ]
    }
  },
  "content/chapter-03/questions/expectiminimax.yml": {
    "type": "yaml",
    "module": {
      "id": "m3-expectiminimax",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Implement expectiminimax with CHANCE nodes for stochastic games.**\n\nHandle MAX, MIN, and CHANCE node types.\n",
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
  "content/chapter-03/questions/minimax-alphabeta.yml": {
    "type": "yaml",
    "module": {
      "id": "m3-minimax-alphabeta",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Implement Minimax with Alpha-Beta pruning for game trees.**\n\n**Level 0** Why do we need special algorithms for games? How does adversarial search differ from single-agent search?\n\n**Level 1** Explain minimax: alternating MAX and MIN layers, utility propagation, and best move selection. Then explain alpha-beta pruning. Provide pseudocode.\n\n**Level 2** Implement minimax decision and alpha-beta pruning for tic-tac-toe or a simple game tree.\n\n**Level 3** Prove alpha-beta's correctness and analyze its complexity. Discuss move ordering's impact on pruning effectiveness.\n",
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
  "content/chapter-03/questions/monte-carlo.yml": {
    "type": "yaml",
    "module": {
      "id": "m3-monte-carlo",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Implement Monte Carlo tree search (MCTS) for game playing.**\n\n**Level 0** Why use random sampling for games? When is MCTS better than minimax?\n\n**Level 1** Explain MCTS phases: selection, expansion, simulation (rollout), backpropagation. Describe how random playouts estimate move quality.\n\n**Level 2** Implement a simple Monte Carlo move selection using random rollouts.\n\n**Level 3** Discuss UCB1 formula, exploration-exploitation balance, and why MCTS excels in large branching factor games like Go.\n",
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
  "content/chapter-03/questions/nash-equilibrium.yml": {
    "type": "yaml",
    "module": {
      "id": "m3-nash-equilibrium",
      "type": "short-answer",
      "chapter": 3,
      "question": "**Find pure Nash equilibria in a 2×2 game.**\n\nGiven payoff matrices A (row player) and B (col player), find all pure Nash equilibria.\n",
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
  "content/chapter-04/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Constraint Satisfaction Problems",
          "concepts": [
            {
              "name": "CSP Representation",
              "description": "Variables, domains, and constraints",
              "exam_questions": [
                "questions/csp-representation.yml"
              ]
            },
            {
              "name": "Backtracking Search",
              "description": "Systematic search with inference for CSPs",
              "exam_questions": [
                "questions/csp-backtracking.yml"
              ]
            },
            {
              "name": "Arc Consistency",
              "description": "Constraint propagation using AC-3",
              "exam_questions": [
                "questions/ac3-algorithm.yml"
              ]
            },
            {
              "name": "Tree-Structured CSPs",
              "description": "Efficient algorithms for tree-structured problems",
              "exam_questions": [
                "questions/tree-csp.yml"
              ]
            },
            {
              "name": "Advanced CSP Techniques",
              "description": "Cutset conditioning and decomposition",
              "exam_questions": [
                "questions/cutset-conditioning.yml"
              ]
            }
          ]
        }
      ]
    }
  },
  "content/chapter-04/questions/ac3-algorithm.yml": {
    "type": "yaml",
    "module": {
      "id": "m4-ac3-algorithm",
      "type": "short-answer",
      "chapter": 4,
      "question": "**Implement the AC-3 arc consistency algorithm.**\n\n**Level 0** What is constraint propagation and why is it powerful for CSPs?\n\n**Level 1** Explain arc consistency: when is an arc (X,Y) consistent? Describe the AC-3 algorithm with queue of arcs. Provide pseudocode.\n\n**Level 2** Implement AC-3 with the REVISE function for a specific constraint type (e.g., not-equal for map coloring).\n\n**Level 3** Analyze AC-3's time complexity O(n²d³) and its role as preprocessing. Discuss stronger consistency levels (path consistency, k-consistency).\n",
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
  "content/chapter-04/questions/csp-backtracking.yml": {
    "type": "yaml",
    "module": {
      "id": "m4-csp-backtracking",
      "type": "short-answer",
      "chapter": 4,
      "question": "**Implement backtracking search with forward checking for CSPs.**\n\n**Level 0** Why are CSPs a useful problem formulation? How do they differ from standard search problems?\n\n**Level 1** Explain CSP components (variables, domains, constraints) and backtracking search. Describe forward checking. Provide pseudocode.\n\n**Level 2** Implement backtracking with forward checking for map coloring or Sudoku.\n\n**Level 3** Analyze heuristics: MRV (minimum remaining values), degree heuristic, least-constraining-value. Discuss time complexity improvements.\n",
      "answer": "Backtracking search assigns variables one at a time, checking consistency at each step. Forward checking prunes future variable domains after each assignment by removing values that conflict with the assignment. When a domain becomes empty, backtrack. For map coloring with k colors and n regions: without inference O(k^n), with forward checking much faster in practice.",
      "topics": [
        "CSP",
        "Backtracking",
        "Forward Checking"
      ],
      "vocab_answer": [
        {
          "word": "domain",
          "definition": "The set of possible values for a variable"
        },
        {
          "word": "constraint",
          "definition": "A restriction on which combinations of values are allowed"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=0K_00aimUEw"
      ]
    }
  },
  "content/chapter-04/questions/csp-representation.yml": {
    "type": "yaml",
    "module": {
      "id": "m4-csp-representation",
      "type": "short-answer",
      "chapter": 4,
      "question": "**Create a CSP data structure and initialize map-coloring.**\n\nWrite MAKE_MAP_COLORING_CSP(regions, edges, colors).\n",
      "answer": "A CSP consists of variables (regions), domains (available colors for each), and constraints (neighboring regions must have different colors). Store this as a structure with neighbor lists.\n\nPseudocode:\n```\nfunction MAKE_MAP_COLORING_CSP(regions, edges, colors):\n    csp = {}\n    csp.vars = regions\n    csp.dom = map()\n    for r in regions: csp.dom[r] = copy(colors)\n    csp.neigh = map(default=[])\n    for (u,v) in edges:\n        csp.neigh[u].append(v)\n        csp.neigh[v].append(u)\n    return csp\n```\n",
      "topics": [
        "CSP",
        "Map Coloring",
        "Problem Representation"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=hJ-6Ma1veUE"
      ]
    }
  },
  "content/chapter-04/questions/cutset-conditioning.yml": {
    "type": "yaml",
    "module": {
      "id": "m4-cutset-conditioning",
      "type": "short-answer",
      "chapter": 4,
      "question": "**Implement cutset conditioning for CSPs.**\n\nChoose cutset variables, enumerate assignments, solve remaining as tree CSP.\n",
      "answer": "Cutset conditioning works by instantiating a set of variables (cutset) such that removing them makes the constraint graph a tree. Then solve the tree CSP for each cutset assignment.\n\nPseudocode:\n```\nfunction CUTSET_CONDITIONING(csp, cutsetVars):\n    for cutAssign in ALL_ASSIGNMENTS(cutsetVars, csp.dom):\n        csp2 = COPY_CSP(csp)\n        if not APPLY_ASSIGNMENT(csp2, cutAssign): continue\n        if not AC3(csp2): continue\n        \n        # assume removing cutset makes graph a tree\n        root = any var not in cutsetVars\n        sol = TREE_CSP_SOLVE(csp2, root)\n        if sol != failure:\n            return MERGE(sol, cutAssign)\n    return failure\n```\n",
      "topics": [
        "Cutset Conditioning",
        "CSP",
        "Tree Decomposition"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=hJ-6Ma1veUE"
      ]
    }
  },
  "content/chapter-04/questions/tree-csp.yml": {
    "type": "yaml",
    "module": {
      "id": "m4-tree-csp",
      "type": "short-answer",
      "chapter": 4,
      "question": "**Solve tree-structured CSP using dynamic programming.**\n\nImplement TREE_CSP_SOLVE(csp, root) with topological ordering.\n",
      "answer": "Tree-structured CSPs can be solved in linear time by making the CSP arc-consistent from leaves to root, then assigning values from root to leaves ensuring consistency with parent assignments.\n\nPseudocode:\n```\nfunction TREE_CSP_SOLVE(csp, root):\n    order = TOPOLOGICAL_ORDER_FROM_ROOT(csp, root)\n    parent = PARENTS_FROM_ROOT(csp, root)\n    \n    # make arc-consistent from leaves upward\n    for X in reverse(order):\n        for Y in csp.neigh[X]:\n            if parent[X] == Y:\n                REVISE(csp, Y, X)\n    \n    assignment = {}\n    assignment[root] = any value in csp.dom[root]\n    for X in order:\n        if X == root: continue\n        P = parent[X]\n        assignment[X] = any v in csp.dom[X] with CONSTRAINT_OK(P, assignment[P], X, v)\n    return assignment\n```\n",
      "topics": [
        "Tree-Structured CSP",
        "Dynamic Programming",
        "Arc Consistency"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=hJ-6Ma1veUE"
      ]
    }
  },
  "content/chapter-05/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Logic and Inference",
          "concepts": [
            {
              "name": "Unification",
              "description": "Pattern matching for logical terms",
              "exam_questions": [
                "questions/unification.yml"
              ]
            },
            {
              "name": "Inference Rules",
              "description": "Forward and backward chaining",
              "exam_questions": [
                "questions/forward-backward-chain.yml",
                "questions/backward-chaining.yml"
              ]
            },
            {
              "name": "Resolution",
              "description": "Theorem proving via resolution",
              "exam_questions": [
                "questions/resolution.yml"
              ]
            },
            {
              "name": "Inference as Search",
              "description": "Treating inference as state-space search",
              "exam_questions": [
                "questions/inference-search.yml"
              ]
            }
          ]
        }
      ]
    }
  },
  "content/chapter-05/questions/backward-chaining.yml": {
    "type": "yaml",
    "module": {
      "id": "m5-backward-chaining",
      "type": "short-answer",
      "chapter": 5,
      "question": "**Implement backward chaining for Horn clauses (goal-driven reasoning).**\n\nWrite BACKWARD_CHAIN(rules, facts, goal, visited).\n",
      "answer": "Backward chaining starts with a goal and works backwards, trying to prove premises of rules that conclude the goal. It's goal-driven and efficient for answering single queries.\n\nPseudocode:\n```\nfunction BACKWARD_CHAIN(rules, facts, goal, visited):\n    if goal in facts: return true\n    if goal in visited: return false\n    visited.add(goal)\n    \n    for r in rules where r.conclusion == goal:\n        ok = true\n        for prem in r.premises:\n            if not BACKWARD_CHAIN(rules, facts, prem, visited):\n                ok = false\n                break\n        if ok: return true\n    return false\n```\n",
      "topics": [
        "Backward Chaining",
        "Logic",
        "Goal-Driven Reasoning"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=vPRDN1Y8kHg"
      ]
    }
  },
  "content/chapter-05/questions/forward-backward-chain.yml": {
    "type": "yaml",
    "module": {
      "id": "m5-forward-backward-chain",
      "type": "short-answer",
      "chapter": 5,
      "question": "**Implement forward and backward chaining for Horn clauses.**\n\n**Level 0** Compare data-driven vs. goal-driven reasoning. When is each approach preferable?\n\n**Level 1** Explain forward chaining (start from facts, apply rules forward) and backward chaining (start from query, work backwards). Provide pseudocode for both.\n\n**Level 2** Implement both algorithms for a simple Horn clause knowledge base.\n\n**Level 3** Analyze completeness, soundness, and complexity. Discuss applications in expert systems and Prolog.\n",
      "answer": "Forward chaining: maintain agenda of known facts, repeatedly select a fact, find rules whose premises are satisfied, add conclusions to agenda. Stops when query is derived or agenda empties. Backward chaining: start with query goal, find rules that conclude it, recursively prove premises. Both are sound and complete for Horn clauses. Forward chaining is data-driven (good for many queries), backward is goal-driven (efficient for single query).",
      "topics": [
        "Forward Chaining",
        "Backward Chaining",
        "Horn Clauses"
      ],
      "vocab_answer": [
        {
          "word": "Horn clause",
          "definition": "A logical clause with at most one positive literal"
        },
        {
          "word": "data-driven",
          "definition": "Reasoning that starts from known facts and derives conclusions"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=vPRDN1Y8kHg"
      ]
    }
  },
  "content/chapter-05/questions/inference-search.yml": {
    "type": "yaml",
    "module": {
      "id": "m5-inference-search",
      "type": "short-answer",
      "chapter": 5,
      "question": "**Implement inference as BFS over derived facts.**\n\nWrite PROOF_BFS(rules, facts, query) that treats proof search as state-space search.\n",
      "answer": "Inference can be formulated as a search problem where states are sets of derived facts, actions are rule applications, and the goal is deriving the query.\n\nPseudocode:\n```\nfunction PROOF_BFS(rules, facts, query):\n    frontier = queue()\n    visitedFacts = set(facts)\n    frontier.enqueue(facts)\n    \n    while not frontier.isEmpty():\n        currentFacts = frontier.dequeue()\n        if query in currentFacts: return true\n        \n        for r in rules:\n            if all prem in currentFacts:\n                nextFacts = currentFacts ∪ {r.conclusion}\n                if r.conclusion not in visitedFacts:\n                    visitedFacts.add(r.conclusion)\n                    frontier.enqueue(nextFacts)\n    return false\n```\n",
      "topics": [
        "Inference",
        "Search",
        "Logic"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=KiCBXu4P-2Y"
      ]
    }
  },
  "content/chapter-05/questions/resolution.yml": {
    "type": "yaml",
    "module": {
      "id": "m5-resolution",
      "type": "short-answer",
      "chapter": 5,
      "question": "**Implement resolution for propositional logic.**\n\nProve KB ⊨ q by refutation using resolution.\n",
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
  "content/chapter-05/questions/unification.yml": {
    "type": "yaml",
    "module": {
      "id": "m5-unification",
      "type": "short-answer",
      "chapter": 5,
      "question": "**Implement the unification algorithm for first-order logic.**\n\n**Level 0** Why is pattern matching essential for logical inference? How does unification enable theorem proving?\n\n**Level 1** Explain unification: finding substitutions that make expressions identical. Handle variables, constants, and functions. Provide pseudocode.\n\n**Level 2** Implement UNIFY(x, y, subst) with occurs check to prevent infinite structures.\n\n**Level 3** Discuss unification's role in resolution, Prolog, and type inference. Analyze time complexity and most general unifiers (MGU).\n",
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
  "content/chapter-06/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Probabilistic Reasoning and Learning",
          "concepts": [
            {
              "name": "Bayesian Networks",
              "description": "Probabilistic graphical models",
              "exam_questions": [
                "questions/bayes-net-inference.yml",
                "questions/likelihood-weighting.yml"
              ]
            },
            {
              "name": "Sequential Models",
              "description": "HMMs and temporal reasoning",
              "exam_questions": [
                "questions/hmm-filtering.yml",
                "questions/particle-filtering.yml"
              ]
            },
            {
              "name": "Machine Learning Basics",
              "description": "Supervised learning fundamentals",
              "exam_questions": [
                "questions/supervised-learning.yml"
              ]
            }
          ]
        }
      ]
    }
  },
  "content/chapter-06/questions/bayes-net-inference.yml": {
    "type": "yaml",
    "module": {
      "id": "m6-bayes-net-inference",
      "type": "short-answer",
      "chapter": 6,
      "question": "**Implement exact inference in Bayesian networks using enumeration.**\n\n**Level 0** Why model uncertainty with probability? How do Bayesian networks compactly represent joint distributions?\n\n**Level 1** Explain Bayes nets: nodes as variables, edges as dependencies, CPTs (conditional probability tables). Describe enumeration algorithm for computing P(X|e). Provide pseudocode.\n\n**Level 2** Implement ENUM_ASK and ENUM_ALL for a simple Bayesian network.\n\n**Level 3** Discuss computational complexity, variable elimination improvements, and approximate inference (likelihood weighting, MCMC).\n",
      "answer": "Enumeration computes P(X|e) by summing joint probabilities over all possible values of hidden variables. ENUM_ASK normalizes the distribution. ENUM_ALL recursively multiplies CPT entries: if variable is observed, use its value; otherwise sum over all possible values. Time complexity exponential in number of variables, but variable elimination reduces it to exponential in tree-width.",
      "topics": [
        "Bayesian Networks",
        "Probabilistic Inference",
        "Graphical Models"
      ],
      "vocab_answer": [
        {
          "word": "CPT",
          "definition": "Conditional Probability Table defining P(X|Parents(X))"
        },
        {
          "word": "evidence",
          "definition": "Observed variable values"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=G-zirzQFWmk"
      ]
    }
  },
  "content/chapter-06/questions/hmm-filtering.yml": {
    "type": "yaml",
    "module": {
      "id": "m6-hmm-filtering",
      "type": "short-answer",
      "chapter": 6,
      "question": "**Implement the forward algorithm for HMM filtering.**\n\n**Level 0** Why model temporal processes with HMMs? What is the filtering task?\n\n**Level 1** Explain HMM components: states, transition model, observation model. Describe filtering: computing P(X_t|e_{1:t}). Provide the recursive forward algorithm.\n\n**Level 2** Implement HMM filtering with belief update using transition and observation matrices.\n\n**Level 3** Discuss Viterbi algorithm for most likely sequence, particle filtering for non-linear models, and applications in speech/robotics.\n",
      "answer": "HMM filtering maintains belief state over hidden variables. Forward algorithm: belief_{t} = α * O_t * T^T * belief_{t-1}, where T is transition matrix, O_t is observation matrix (diagonal), α normalizes. This is a Bayesian update: prediction (apply transition) then correction (incorporate observation). Time O(n²) per step for n states.",
      "topics": [
        "Hidden Markov Models",
        "Filtering",
        "Temporal Reasoning"
      ],
      "vocab_answer": [
        {
          "word": "filtering",
          "definition": "Estimating current state from past observations"
        },
        {
          "word": "belief state",
          "definition": "Probability distribution over possible hidden states"
        }
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=kqSzLo9fenk"
      ]
    }
  },
  "content/chapter-06/questions/likelihood-weighting.yml": {
    "type": "yaml",
    "module": {
      "id": "m6-likelihood-weighting",
      "type": "short-answer",
      "chapter": 6,
      "question": "**Approximate P(X|e) using likelihood weighting.**\n\nImplement LIKELIHOOD_WEIGHTING and WEIGHTED_SAMPLE.\n",
      "answer": "Likelihood weighting is an approximate inference method that generates samples consistent with evidence by fixing observed variables and weighting samples by the probability of the evidence.\n\nPseudocode:\n```\nfunction LIKELIHOOD_WEIGHTING(X, e, bn, N):\n    W = map(value -> 0)\n    for i in 1..N:\n        (sample, w) = WEIGHTED_SAMPLE(bn, e)\n        W[sample[X]] += w\n    return NORMALIZE(W)\n\nfunction WEIGHTED_SAMPLE(bn, e):\n    w = 1\n    sample = copy(e)\n    for Y in bn.topoOrder:\n        if Y in e:\n            w *= P(Y=e[Y] | parents(Y)=sample[parents(Y)], bn)\n        else:\n            sample[Y] = SAMPLE_FROM(P(Y | parents(Y)=sample[parents(Y)], bn))\n    return (sample, w)\n```\n",
      "topics": [
        "Likelihood Weighting",
        "Approximate Inference",
        "Bayesian Networks"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=G-zirzQFWmk"
      ]
    }
  },
  "content/chapter-06/questions/particle-filtering.yml": {
    "type": "yaml",
    "module": {
      "id": "m6-particle-filtering",
      "type": "short-answer",
      "chapter": 6,
      "question": "**Implement particle filtering for dynamic Bayesian networks.**\n\nWrite PARTICLE_FILTER with propagate, weight, and resample steps.\n",
      "answer": "Particle filtering approximates belief distributions using weighted samples (particles). Each step: propagate particles through transition model, weight by observation likelihood, resample based on weights.\n\nPseudocode:\n```\nfunction PARTICLE_FILTER(particles, observation, TRANSITION, OBS_LIK, N):\n    # 1) propagate\n    for i in 1..N:\n        particles[i] = TRANSITION(particles[i])\n    \n    # 2) weight by observation likelihood\n    weights = []\n    for i in 1..N:\n        weights[i] = OBS_LIK(observation, particles[i])\n    \n    # 3) resample\n    particles = RESAMPLE(particles, weights, N)\n    return particles\n```\n",
      "topics": [
        "Particle Filtering",
        "Sequential Monte Carlo",
        "Bayesian Filtering"
      ],
      "example_videos": [
        "https://www.youtube.com/watch?v=aUkBa1zMKv4"
      ]
    }
  },
  "content/chapter-06/questions/supervised-learning.yml": {
    "type": "yaml",
    "module": {
      "id": "m6-supervised-learning",
      "type": "short-answer",
      "chapter": 6,
      "question": "**Implement a supervised learning training pipeline with train/test split.**\n\nWrite TRAIN_EVAL(dataset, model, lossFn, optimizer, epochs).\n",
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
  "content/chapter-07/concept-map.yml": {
    "type": "yaml",
    "module": {
      "concept_map": [
        {
          "category": "Neural Networks and Transformers",
          "concepts": [
            {
              "name": "Feedforward Networks",
              "description": "Basic neural networks and backpropagation",
              "exam_questions": [
                "questions/feedforward-backprop.yml"
              ]
            },
            {
              "name": "Convolutional Networks",
              "description": "CNNs for computer vision",
              "exam_questions": [
                "questions/convolution.yml"
              ]
            },
            {
              "name": "Attention and Transformers",
              "description": "Self-attention mechanisms",
              "exam_questions": [
                "questions/transformer-attention.yml",
                "questions/transformer-block.yml"
              ]
            },
            {
              "name": "Language Models",
              "description": "Training next-token prediction models",
              "exam_questions": [
                "questions/language-model-training.yml"
              ]
            }
          ]
        }
      ]
    }
  },
  "content/chapter-07/questions/convolution.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-convolution",
      "type": "short-answer",
      "chapter": 7,
      "question": "**Implement a single 2D convolution forward pass (no padding, stride 1).**\n\nWrite CONV2D(image, kernel).\n",
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
  "content/chapter-07/questions/feedforward-backprop.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-feedforward-backprop",
      "type": "short-answer",
      "chapter": 7,
      "question": "**Implement forward and backward passes for a 2-layer neural network.**\n\n**Level 0** Why are neural networks so powerful for learning? What does \"learning\" mean mathematically?\n\n**Level 1** Explain feedforward computation (linear + activation layers) and backpropagation (chain rule for gradients). Describe one training step with loss calculation and weight updates.\n\n**Level 2** Implement a training step for 2-layer MLP with ReLU activation and softmax output.\n\n**Level 3** Derive backpropagation equations from chain rule. Discuss gradient descent variants (SGD, Adam), initialization, and universal approximation theorem.\n",
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
  "content/chapter-07/questions/language-model-training.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-language-model-training",
      "type": "short-answer",
      "chapter": 7,
      "question": "**Write pseudocode to train a next-token transformer language model.**\n\nImplement TRAIN_LM(corpusText, tokenizer, model, optimizer, steps, seqLen, batchSize).\n",
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
  "content/chapter-07/questions/transformer-attention.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-transformer-attention",
      "type": "short-answer",
      "chapter": 7,
      "question": "**Implement self-attention mechanism and explain transformers.**\n\n**Level 0** Why did attention mechanisms revolutionize NLP and AI? What problem do they solve?\n\n**Level 1** Explain self-attention: Query, Key, Value matrices, scaled dot-product, and weighted sum. Describe transformer blocks with multi-head attention and feedforward layers.\n\n**Level 2** Implement single-head attention: scores = QK^T/sqrt(d), weights = softmax(scores), output = weights * V.\n\n**Level 3** Discuss multi-head attention, positional encoding, layer normalization, and training language models. Explain why transformers scale better than RNNs.\n",
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
  "content/chapter-07/questions/transformer-block.yml": {
    "type": "yaml",
    "module": {
      "id": "m7-transformer-block",
      "type": "short-answer",
      "chapter": 7,
      "question": "**Implement a transformer block forward pass with residual connections and layer normalization.**\n\nWrite TRANSFORMER_BLOCK(X, params).\n",
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
  "content/chapter-01/index.md": {
    "type": "markdown",
    "module": "# Module 1: Intelligent Agents and Search\r\n\r\nThis module introduces the fundamental concepts of intelligent agents and explores various search algorithms that enable agents to find solutions in problem spaces.\r\n\r\n## Topics Covered\r\n\r\n### Intelligent Agents\r\n- What makes an agent rational?\r\n- Different types of agent architectures\r\n- Performance measures and environment types\r\n\r\n### Uninformed Search Strategies\r\n- Breadth-First Search (BFS)\r\n- Depth-First Search (DFS)\r\n- Uniform Cost Search (UCS)\r\n- Maze navigation problems\r\n\r\n### Informed Search Strategies\r\n- Heuristic functions\r\n- A* Search\r\n- Greedy Best-First Search\r\n- Admissibility and consistency\r\n\r\n## Practice Questions\r\n\r\nWork through the practice questions in this module to test your understanding of intelligent agents and search algorithms.\r\n"
  },
  "content/chapter-02/index.md": {
    "type": "markdown",
    "module": "# Module 2: Local Search and Optimization\r\n\r\nThis module explores local search algorithms that navigate through the space of complete states, improving solutions iteratively rather than building paths from an initial state.\r\n\r\n## Topics Covered\r\n\r\n### Local Search Fundamentals\r\n- Objective functions and optimization landscapes\r\n- Hill climbing and local optima\r\n- The exploration vs. exploitation trade-off\r\n\r\n### Stochastic Methods\r\n- Simulated annealing\r\n- Genetic algorithms\r\n- Random search techniques\r\n\r\n### Applications\r\n- N-Queens problem\r\n- Continuous optimization\r\n- Real-world constraint satisfaction\r\n\r\n## Practice Questions\r\n\r\nWork through the practice questions in this module to understand local search and optimization algorithms.\r\n"
  },
  "content/chapter-03/index.md": {
    "type": "markdown",
    "module": "# Module 3: Adversarial Search and Games\r\n\r\nGame playing with intelligent agents competing against each other.\r\n\r\n## Topics\r\n\r\n- Minimax algorithm\r\n- Alpha-beta pruning\r\n- Monte Carlo tree search\r\n- Game theory basics\r\n\r\n## Practice Questions\r\n\r\nExplore adversarial search through practice problems.\r\n"
  },
  "content/chapter-04/index.md": {
    "type": "markdown",
    "module": "# Module 4: Constraint Satisfaction Problems\r\n\r\nSolving problems by finding assignments that satisfy constraints.\r\n\r\n## Topics\r\n\r\n- CSP formulation\r\n- Backtracking search\r\n- Arc consistency (AC-3)\r\n- Tree-structured CSPs\r\n\r\n## Practice Questions\r\n\r\nMaster CSP algorithms through practice.\r\n"
  },
  "content/chapter-05/index.md": {
    "type": "markdown",
    "module": "# Module 5: Logic and Inference\r\n\r\nUsing logical reasoning for knowledge representation and automated theorem proving.\r\n\r\n## Topics\r\n\r\n- First-order logic\r\n- Unification\r\n- Forward and backward chaining\r\n- Resolution\r\n\r\n## Practice Questions\r\n\r\nPractice logical inference and automated reasoning.\r\n"
  },
  "content/chapter-06/index.md": {
    "type": "markdown",
    "module": "# Module 6: Probabilistic Reasoning and Learning\r\n\r\nReasoning under uncertainty using probability and Bayesian networks.\r\n\r\n## Topics\r\n\r\n- Bayesian networks\r\n- Exact and approximate inference\r\n- Hidden Markov Models\r\n- Particle filtering\r\n- Supervised learning basics\r\n\r\n## Practice Questions\r\n\r\nExplore probabilistic reasoning techniques.\r\n"
  },
  "content/chapter-07/index.md": {
    "type": "markdown",
    "module": "# Module 7: Neural Networks and Transformers\r\n\r\nDeep learning fundamentals and modern architectures.\r\n\r\n## Topics\r\n\r\n- Feedforward networks and backpropagation\r\n- Convolutional networks\r\n- Attention mechanisms\r\n- Transformer architecture\r\n- Language model training\r\n\r\n## Practice Questions\r\n\r\nUnderstand deep learning through implementation.\r\n"
  },
  "index.md": {
    "type": "markdown",
    "module": "# CS440: Artificial Intelligence\r\n\r\nWelcome to CS440! This course explores the foundational concepts and techniques of artificial intelligence, from intelligent agents and search algorithms to machine learning and beyond.\r\n\r\n## Course Overview\r\n\r\nThis course provides a comprehensive introduction to artificial intelligence, covering fundamental concepts, algorithms, and paradigms that enable machines to exhibit intelligent behavior. Through hands-on problems and theoretical exploration, students will master core AI techniques including search algorithms, knowledge representation, planning, reasoning under uncertainty, and machine learning fundamentals.\r\n\r\n## Topics\r\n\r\n### [Module 1: Intelligent Agents and Search](content/chapter-01/)\r\n\r\n**Problem Formulation:**\r\n* State spaces and problem representation\r\n* 8-puzzle formulation\r\n\r\n**Uninformed Search:**\r\n* Breadth-First Search (BFS)\r\n* Iterative Deepening DFS (IDDFS)\r\n* Maze navigation algorithms\r\n\r\n**Informed Search:**\r\n* A* search with admissible heuristics\r\n* Heuristic design and evaluation\r\n\r\n---\r\n\r\n### [Module 2: Local Search and Optimization](content/chapter-02/)\r\n\r\n**Hill Climbing:**\r\n* 8-Queens problem\r\n* Local optima and plateaus\r\n\r\n**Stochastic Methods:**\r\n* Simulated annealing\r\n* Genetic algorithms\r\n* Continuous optimization\r\n\r\n---\r\n\r\n### [Module 3: Adversarial Search and Games](content/chapter-03/)\r\n\r\n**Game Playing:**\r\n* Minimax algorithm\r\n* Alpha-beta pruning\r\n* Game tree evaluation\r\n\r\n**Stochastic Games:**\r\n* Monte Carlo tree search (MCTS)\r\n* Random rollouts\r\n* Nash equilibrium basics\r\n\r\n---\r\n\r\n### [Module 4: Constraint Satisfaction Problems](content/chapter-04/)\r\n\r\n**CSP Formulation:**\r\n* Variables, domains, and constraints\r\n* Map coloring and Sudoku\r\n\r\n**Search and Inference:**\r\n* Backtracking with forward checking\r\n* Arc consistency (AC-3)\r\n* Tree-structured CSPs\r\n\r\n---\r\n\r\n### [Module 5: Logic and Inference](content/chapter-05/)\r\n\r\n**First-Order Logic:**\r\n* Unification algorithm\r\n* Term matching and substitution\r\n\r\n**Inference Methods:**\r\n* Forward chaining\r\n* Backward chaining\r\n* Resolution\r\n\r\n---\r\n\r\n### [Module 6: Probabilistic Reasoning and Learning](content/chapter-06/)\r\n\r\n**Bayesian Networks:**\r\n* Exact inference with enumeration\r\n* Likelihood weighting\r\n\r\n**Temporal Models:**\r\n* Hidden Markov Models (HMMs)\r\n* Forward algorithm (filtering)\r\n* Particle filtering\r\n\r\n**Supervised Learning:**\r\n* Training and evaluation\r\n* Loss functions and optimization\r\n\r\n---\r\n\r\n### [Module 7: Neural Networks and Transformers](content/chapter-07/)\r\n\r\n**Deep Learning Fundamentals:**\r\n* Feedforward networks\r\n* Backpropagation\r\n* Convolutional networks\r\n\r\n**Modern Architectures:**\r\n* Self-attention mechanism\r\n* Transformer blocks\r\n* Language model training\r\n\r\n---\r\n\r\n## Getting Started\r\n\r\nNavigate to the module content using the navigation panel on the left. Each module contains:\r\n\r\n- **Concept explanations** with multiple complexity levels\r\n- **Practice questions** to test your understanding\r\n- **Video resources** for visual learners\r\n- **Hands-on coding exercises** to implement AI algorithms\r\n\r\nGood luck with your AI journey!\r\n"
  }
};

export const stats = {
  yamlCount: 42,
  markdownCount: 8,
  totalCount: 50
};
