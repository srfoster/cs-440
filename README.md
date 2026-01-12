# cs-440

TODO:
* Programming env and lang?  https://colab.research.google.com/drive/1HeQsvxqIZkdb6tpIpkDW1Bfes9cmOZVN#scrollTo=4682551b
* Collab for the boring stuff and solara for cool sims?


QUESTIONS

Chapter 1 — Agents, State Spaces, Classical Search
1) 8-puzzle: ACTIONS / RESULT / GOAL_TEST

Exam prompt (human-readable):
The 8-puzzle is a classic AI search problem consisting of a 3×3 grid containing tiles numbered 1–8 and one empty space (the blank). A state is a particular arrangement of the tiles. An action is a legal move of the blank (up/down/left/right), and the result of an action is the new state produced by swapping the blank with the neighboring tile. A goal test is the procedure that checks whether a state is the goal configuration (usually [[1,2,3],[4,5,6],[7,8,0]]). This problem matters because it is one of the simplest settings where students can demonstrate the core idea of problem formulation: before an algorithm can solve something, you must define the state space precisely.
Task: Write pseudocode for ACTIONS(s), RESULT(s,a), and GOAL_TEST(s) for 8-puzzle states represented as a 3×3 grid with 0 as the blank.

2) BFS and IDDFS skeletons

Exam prompt (human-readable):
Many AI problems can be framed as state-space search, where we explore a graph whose nodes are states and whose edges are actions. Breadth-first search (BFS) explores states in increasing order of depth from the start; it is complete (will find a solution if one exists) and optimal for equal step costs, but often uses large memory. Iterative deepening depth-first search (IDDFS) combines the low memory footprint of DFS with the completeness/optimality (in unit-cost settings) of BFS by repeatedly running depth-limited DFS with increasing depth limits. These matter because they are foundational “default tools” and because the tradeoffs (time vs memory, optimality vs practicality) appear across all of AI.
Task: Given the standard search interface ACTIONS(s), RESULT(s,a), GOAL_TEST(s), write pseudocode for BFS(start) and IDDFS(start, maxDepth). Your BFS should avoid revisiting states; your IDDFS should show a depth loop and a depth-limited search helper.

3) A* search

Exam prompt (human-readable):
A* (pronounced “A-star”) is a best-first search algorithm that uses an evaluation function

f(s)=g(s)+h(s)
f(s)=g(s)+h(s)

where g(s) is the known cost from the start to state s, and h(s) is a heuristic: an estimate of the remaining cost from s to the goal. A* is widely used in navigation and planning because, with an admissible heuristic (never overestimates true remaining cost), it is complete and optimal, and is often dramatically faster than uniform-cost search in practice. This matters because A* is the canonical example of how “good modeling” (a heuristic) can radically change performance while preserving correctness.
Task: Write pseudocode for A_STAR(start, h) for maze navigation using a priority queue ordered by f=g+h. Your implementation should maintain g-costs and support “decrease-key” behavior when a better path to a state is found.

4) Depth-limited DFS cutoff

Exam prompt (human-readable):
A depth-limited search is a variant of DFS that explores paths only up to a maximum depth. A cutoff is used when deeper search becomes too expensive, when we only care about short solutions, or when we want a controlled approximation of DFS (and as a building block for IDDFS). This matters because real AI systems often cannot afford unlimited search; they must manage computation using resource bounds while still behaving sensibly.
Task: Write pseudocode for DFS_CUTOFF(start, cutoff) that performs DFS but stops expanding nodes beyond the depth limit. The function should return a solution path if found within the cutoff depth, and failure otherwise.

5) Boids: one simulation tick

Exam prompt (human-readable):
A multi-agent system is a system composed of multiple interacting agents whose collective behavior may be complex even if each agent’s rules are simple. The boids model is a classic example: each boid updates its velocity based on local neighbors using three intuitive rules—separation (avoid crowding), alignment (match heading), and cohesion (move toward neighbors). This matters because it shows a key AI theme: intelligent-seeming global behavior can emerge from decentralized local rules, and it contrasts sharply with search-based AI.
Task: Write pseudocode for a function STEP_BOIDS(boids, params) that computes one synchronous update step (all boids updated based on the old configuration), applying weighted separation/alignment/cohesion forces and a speed cap.

Chapter 2 — Local Search and Optimization
1) Objective function for 8-queens

Exam prompt (human-readable):
In local search, we do not build solution paths; instead we treat problem-solving as optimization. We define an objective function that scores candidate solutions, and we search for a state with a better score. The 8-queens problem can be approached this way by defining an objective like “number of conflicting queen pairs,” which we want to minimize. This matters because many real-world problems are naturally “optimize a score” rather than “reach a goal state through a shortest path.”
Task: Using the representation board[col] = row, write pseudocode for CONFLICTS(board) that returns the number of attacking queen pairs (same row or diagonal).

2) Hill climbing

Exam prompt (human-readable):
Hill climbing is a greedy local search method: from a current state, examine neighboring states and move to a neighbor with a better objective value. It is simple and often fast, but it can get stuck in local minima/maxima, plateaus, or ridges. This matters because hill climbing is a “first local search idea” that reveals the core limitation of greedy methods and motivates stochastic techniques.
Task: Write pseudocode for HILL_CLIMB(board) that repeatedly chooses the best neighboring board (lowest conflicts) until no improvement is possible, then returns the final board.

3) Simulated annealing

Exam prompt (human-readable):
Simulated annealing is a stochastic optimization method inspired by physical annealing in metallurgy. Unlike hill climbing, it sometimes accepts worse moves early on to escape local minima, controlled by a temperature parameter T. As T decreases according to a cooling schedule, the algorithm becomes less willing to accept worse moves. This matters because it’s one of the cleanest illustrations of how adding randomness can turn a brittle greedy method into something far more robust.
Task: Write pseudocode for SIM_ANNEAL(board, T0, alpha, steps) that selects random neighbors, always accepts improvements, sometimes accepts worse moves with probability exp(-delta/T), and updates temperature each step.

4) Genetic algorithm

Exam prompt (human-readable):
A genetic algorithm (GA) is a population-based optimization technique inspired by biological evolution. Candidates (called individuals) are evaluated by a fitness function, and new individuals are produced via selection, crossover (recombining “genes”), and mutation (random variation). This matters because GAs model an important AI strategy: exploit good solutions while exploring diverse alternatives, and they are useful when gradients or clean heuristics are unavailable.
Task: Write pseudocode for GA_8QUEENS(popSize, generations, mutRate) including: a fitness function derived from conflicts, a selection method, crossover, mutation, and a main loop that returns a best board found.

5) Continuous optimization: airport placement

Exam prompt (human-readable):
Many optimization problems have continuous state spaces (real-valued parameters) rather than discrete states. In an airport placement problem, the airport location (x,y) can be anywhere on a plane, and we define an objective such as the weighted sum of squared distances to towns. Discrete graph search methods are not a natural fit here; instead we use continuous optimization methods like random search, hill climbing in ℝ², or gradient methods. This matters because it forces you to recognize when “state space as a graph” stops being the right mental model.
Task: Write pseudocode for AIRPORT_OBJECTIVE(x,y,towns) and RANDOM_SEARCH_CONTINUOUS(towns, bounds, iters) that samples candidate points and tracks the best solution.

Chapter 3 — Adversarial Search and Games
1) Minimax

Exam prompt (human-readable):
In adversarial search, the environment includes an opponent who actively tries to reduce your success. In a deterministic, turn-based, two-player zero-sum game (like tic-tac-toe), the canonical solution concept is the minimax strategy: assume your opponent plays optimally, and choose moves that maximize your guaranteed outcome. This matters because it introduces the fundamental AI shift from “find a good plan” to “find a plan that is robust against an adversary.”
Task: Write pseudocode for MINIMAX_DECISION(state, player) and the recursive evaluators needed to compute the minimax value using TERMINAL, UTILITY, ACTIONS, and RESULT.

2) Alpha–beta pruning

Exam prompt (human-readable):
Alpha–beta pruning is an optimization to minimax that avoids exploring branches that cannot affect the final decision. Alpha represents the best value the MAX player can guarantee so far; beta represents the best value the MIN player can guarantee so far. When a branch proves worse than what a player can already force, it can be pruned without changing correctness. This matters because it’s a key example of algorithmic speedups that preserve exact optimal play—turning infeasible game trees into feasible ones in practice.
Task: Write pseudocode for ALPHABETA_DECISION including AB_MAX and AB_MIN that correctly uses alpha and beta cutoffs.

3) Expectiminimax

Exam prompt (human-readable):
Some games contain chance events (dice rolls, random draws). In these settings, minimax generalizes to expectiminimax, which handles MAX nodes (your choices), MIN nodes (opponent’s choices), and CHANCE nodes (random outcomes). At chance nodes we compute expected value using the probability of each outcome. This matters because many real decision-making problems mix adversaries and randomness, and you must model both correctly to make rational decisions.
Task: Write pseudocode for EXPECTIMINIMAX(s, player) that branches based on node type and uses an OUTCOMES function at chance nodes.

4) Monte Carlo rollouts

Exam prompt (human-readable):
When the game tree is too large for exact minimax—even with alpha–beta pruning—agents can use Monte Carlo simulation to estimate move quality. A rollout (playout) simulates a complete game to terminal using random (or heuristic) moves, and the outcomes are averaged to estimate expected utility. This matters because Monte Carlo ideas are central in modern game-playing systems (and more broadly in approximate planning under uncertainty).
Task: Write pseudocode for MONTE_CARLO_MOVE(state, player, rolloutsPerMove) and RANDOM_PLAYOUT.

5) Pure Nash equilibria in 2×2

Exam prompt (human-readable):
In game theory, a Nash equilibrium is a strategy profile in which no player can improve their payoff by unilaterally changing their own strategy. For a finite normal-form game, Nash equilibria represent stable outcomes under self-interested behavior. In a 2×2 game, you can test for pure-strategy equilibria by checking whether each player’s chosen action is a best response to the other’s action. This matters because it connects “adversarial reasoning” to broader multi-agent systems where players are not strictly zero-sum.
Task: Given payoff matrices A and B, write pseudocode for PURE_NASH_2x2(A,B) that returns all pure Nash equilibria.

Chapter 4 — CSPs
1) CSP representation

Exam prompt (human-readable):
A constraint satisfaction problem (CSP) is defined by variables, domains (allowed values per variable), and constraints (rules restricting combinations of values). Examples include map coloring and Sudoku. CSPs matter because they show a powerful alternative to generic search: you exploit constraints to prune huge portions of the search space using inference.
Task: Write pseudocode to create a map-coloring CSP structure given regions, adjacency edges, and colors.

2) Backtracking + forward checking

Exam prompt (human-readable):
Backtracking search solves CSPs by assigning variables one at a time and undoing assignments when they lead to a dead end. Forward checking is an inference technique that prunes neighbor domains immediately after an assignment, detecting failures early. This matters because it demonstrates the key CSP theme: solving is not only “search,” but “search + inference,” and inference can produce orders-of-magnitude speedups.
Task: Write pseudocode for BACKTRACK(assignment, csp) that includes forward checking and domain restoration on backtracking.

3) AC-3 arc consistency

Exam prompt (human-readable):
Arc consistency enforces a local constraint property: for every value in a variable’s domain, there must exist a compatible value in each neighbor’s domain (for the constraint between them). The AC-3 algorithm repeatedly applies a REVISE step across arcs until no more pruning is possible. This matters because it’s a canonical inference algorithm and a building block for more advanced CSP solvers; it can dramatically shrink domains before or during search.
Task: Write pseudocode for AC3(csp) and REVISE(csp,X,Y).

4) Tree-structured CSP solver

Exam prompt (human-readable):
The difficulty of CSP solving depends heavily on the structure of the constraint graph. If the constraint graph is a tree (acyclic), the CSP can be solved in time linear in the number of variables (up to domain factors) using dynamic programming: first enforce consistency from leaves to root, then assign values from root to leaves. This matters because it shows that “NP-hard in general” can become “easy” under structural restrictions, motivating concepts like treewidth.
Task: Write pseudocode for TREE_CSP_SOLVE(csp, root) that solves a tree-structured CSP efficiently.

5) Cutset conditioning

Exam prompt (human-readable):
Cutset conditioning is a technique for solving general CSPs by removing a small set of variables (a cutset) so that the remaining constraint graph becomes a tree. You then enumerate assignments to the cutset and solve the remaining tree-structured CSP efficiently for each case. This matters because it makes the key structural idea operational: if treewidth is small, hard problems become tractable in practice.
Task: Write pseudocode for CUTSET_CONDITIONING(csp, cutsetVars) that enumerates cutset assignments, runs inference, and solves the remaining tree.

Chapter 5 — Logic
1) Unification

Exam prompt (human-readable):
In first-order logic, unification is the process of finding a substitution for variables that makes two logical expressions identical. It is fundamental to resolution, theorem proving, and logic programming (e.g., Prolog). This matters because unification is the “mechanical heart” of symbolic reasoning: it’s how a system matches patterns and applies general rules to specific situations.
Task: Write pseudocode for UNIFY(x, y, subst) including variable handling and occurs-check.

2) Forward chaining

Exam prompt (human-readable):
Forward chaining is a data-driven inference method for Horn clauses: starting from known facts, repeatedly apply rules whose premises are satisfied to derive new facts. It matters because it models rule-based expert systems and production systems, and demonstrates a clear form of inference that can be efficient when there are many facts and you want all consequences.
Task: Write pseudocode for FORWARD_CHAIN(rules, facts, query) that returns whether the query is derivable.

3) Backward chaining

Exam prompt (human-readable):
Backward chaining is a goal-driven inference method: start with a query (goal) and work backward by finding rules that could conclude it, recursively trying to prove their premises. This matters because it resembles how humans often reason (“to prove this, what would need to be true?”) and because it is efficient when you only care about a specific query rather than all consequences.
Task: Write pseudocode for BACKWARD_CHAIN(rules, facts, goal, visited) that avoids cycles.

4) Propositional resolution

Exam prompt (human-readable):
Resolution is a single, sound inference rule that can be used (with CNF conversion) to perform refutation proof: to show that a knowledge base entails a query, add the negation of the query and derive a contradiction (the empty clause). This matters because it is one of the most important completeness results in automated reasoning: a simple mechanical rule can, in principle, prove any propositional entailment.
Task: Write pseudocode for RESOLUTION_ENTAILS(KB_clauses, q).

5) Inference as search (BFS)

Exam prompt (human-readable):
Many inference systems can be understood as search: you explore a space of derived statements by repeatedly applying inference rules. This viewpoint matters because it connects symbolic reasoning to the earlier parts of AI (search), clarifying that the difference is often the representation and operators, not the “shape” of the computation.
Task: Write pseudocode for PROOF_BFS(rules, facts, query) that derives new facts by applying rules and returns whether the query is reachable.

Chapter 6 — Probabilistic Reasoning + Learning
1) Exact Bayes net inference by enumeration

Exam prompt (human-readable):
A Bayesian network is a directed acyclic graph where nodes are random variables and edges represent direct probabilistic dependencies. It compactly represents a joint distribution using conditional probability tables (CPTs) and conditional independence structure. Exact inference answers queries like 
P(X∣e)
P(X∣e) by summing out hidden variables, but can be expensive. This matters because Bayes nets are the canonical model for probabilistic reasoning, and enumeration shows the “ground truth” algorithm against which approximations are compared.
Task: Write pseudocode for ENUM_ASK(X, e, bn) and ENUM_ALL(vars, e, bn).

2) Likelihood weighting

Exam prompt (human-readable):
Exact inference in Bayesian networks can be intractable for large networks, so we use approximate inference. Likelihood weighting is a sampling method that handles evidence by fixing evidence variables and weighting samples by how likely that evidence is under the sampled parent values. This matters because it introduces a practical and widely used idea: use randomness plus weighting to approximate probabilities when exact computation is too costly.
Task: Write pseudocode for LIKELIHOOD_WEIGHTING(X, e, bn, N) and its weighted sampling helper.

3) HMM filtering (forward algorithm)

Exam prompt (human-readable):
A Hidden Markov Model (HMM) represents a system with a hidden state that evolves over time according to a Markov transition model, while producing noisy observations. Filtering computes the belief distribution over the current hidden state given all observations so far. This matters because it’s the prototypical “reasoning over time” model, used in tracking, speech, robotics, and many sequential inference tasks.
Task: Write pseudocode for HMM_FILTER(priorBelief, observations, T, O) that performs repeated predict–update–normalize steps.

4) Particle filtering

Exam prompt (human-readable):
When the state space is large or continuous, exact HMM/DBN inference becomes infeasible. Particle filtering approximates the belief state with a set of weighted samples (particles) that are propagated through the dynamics and reweighted by observation likelihood, then resampled. This matters because particle filtering is a standard practical algorithm for robotics and tracking, and it cleanly illustrates approximate inference as “simulation + correction.”
Task: Write pseudocode for PARTICLE_FILTER(particles, observation, TRANSITION, OBS_LIK, N) including propagation, weighting, and resampling.

5) Supervised learning training pipeline

Exam prompt (human-readable):
In supervised learning, we learn a function from inputs to outputs using labeled examples. A typical pipeline includes splitting data into training and test sets, training a model by minimizing a loss function using an optimizer (e.g., gradient descent), and evaluating generalization performance (e.g., accuracy). This matters because it is the core workflow of modern applied machine learning, and students must understand the loop that converts data into a working model.
Task: Write pseudocode for TRAIN_EVAL(dataset, model, lossFn, optimizer, epochs) showing training and evaluation.

Chapter 7 — Neural Nets + Transformers + “Make Your Own ChatGPT”
1) One training step for a 2-layer neural net

Exam prompt (human-readable):
A feedforward neural network composes linear transformations with nonlinear activations to learn complex functions. Backpropagation computes gradients of the loss with respect to parameters efficiently using the chain rule, enabling gradient-based training. This matters because it is the foundational mechanism behind deep learning—understanding a single training step means you understand the computational core of training at any scale.
Task: Write pseudocode for one training step of a 2-layer MLP with ReLU and softmax, computing loss, gradients, and parameter updates.

2) Vision example: convolution forward pass

Exam prompt (human-readable):
A convolution is a local weighted sum applied across an image, producing feature maps that detect patterns such as edges, corners, or textures. Convolutions matter because they capture spatial locality and weight sharing, which are central to classic deep vision models (CNNs) and still relevant for many modern systems. Writing convolution explicitly demonstrates you understand what “a conv layer” actually computes.
Task: Write pseudocode for CONV2D(image, kernel) with stride 1 and no padding.

3) Self-attention (single head)

Exam prompt (human-readable):
Self-attention is the core operation of transformers. Given queries (Q), keys (K), and values (V), attention computes how much each token should “attend” to every other token, producing context-sensitive representations. The scaling factor 
1/d
1/
d

​

 stabilizes gradients when the key/query dimension is large. This matters because attention is the primary reason transformers outperform earlier sequence models on language tasks, and understanding it is essential for understanding LLMs.
Task: Write pseudocode for ATTENTION(Q,K,V) using scaled dot-product attention and row-wise softmax.

4) Transformer block

Exam prompt (human-readable):
A transformer block typically consists of (1) a self-attention sublayer and (2) a position-wise feedforward sublayer, each wrapped with residual connections and layer normalization. Residuals help optimization by preserving gradient flow; layer norm stabilizes activations. This matters because the transformer block is the basic repeating unit of LLMs—understanding one block means you understand the architecture at scale.
Task: Write pseudocode for TRANSFORMER_BLOCK(X, params) showing attention, residual, layer norm, feedforward, residual, and layer norm.

5) Train a tiny next-token language model

Exam prompt (human-readable):
A language model learns to predict the next token given previous tokens. Modern LLMs are trained by taking large text corpora, converting text into tokens with a tokenizer, sampling fixed-length sequences, and training a transformer to minimize cross-entropy loss for next-token prediction. This matters because it is the conceptual backbone of “make your own ChatGPT”: while scaling is expensive, the training loop and objective are straightforward and reveal what the model is actually learning.
Task: Write pseudocode for TRAIN_LM(corpusText, tokenizer, model, optimizer, steps, seqLen, batchSize) that samples sequences, builds shifted targets, computes loss, backpropagates, and updates parameters.



ANSWERS

Chapter 1 — Agents, State Spaces, Classical Search
1) 8-puzzle problem formulation

Q (pseudocode): Write ACTIONS(s), RESULT(s,a), and GOAL_TEST(s) for 8-puzzle state s as a 3×3 grid with 0 as blank.

function ACTIONS(s):
    (r,c) = findBlank(s)          # where s[r][c] == 0
    acts = empty list
    if r > 0: add "UP" to acts
    if r < 2: add "DOWN" to acts
    if c > 0: add "LEFT" to acts
    if c < 2: add "RIGHT" to acts
    return acts

function RESULT(s, a):
    s2 = deepCopy(s)
    (r,c) = findBlank(s2)
    (r2,c2) = (r,c)
    if a == "UP":    r2 = r-1
    if a == "DOWN":  r2 = r+1
    if a == "LEFT":  c2 = c-1
    if a == "RIGHT": c2 = c+1
    swap s2[r][c] and s2[r2][c2]
    return s2

function GOAL_TEST(s):
    goal = [[1,2,3],[4,5,6],[7,8,0]]
    return s == goal

2) BFS vs DFS vs IDDFS vs A*: implement BFS and IDDFS skeletons

Q: Implement BFS(start) and IDDFS(start, maxDepth) over generic state space using ACTIONS/RESULT/GOAL_TEST.

function BFS(start):
    if GOAL_TEST(start): return [start]
    frontier = queue()
    frontier.enqueue(start)
    parent = map()          # parent[state] = (prevState, action)
    parent[start] = null
    visited = set([start])

    while not frontier.isEmpty():
        s = frontier.dequeue()
        for a in ACTIONS(s):
            s2 = RESULT(s,a)
            if s2 not in visited:
                visited.add(s2)
                parent[s2] = (s, a)
                if GOAL_TEST(s2):
                    return RECONSTRUCT_PATH(parent, s2)
                frontier.enqueue(s2)
    return failure

function IDDFS(start, maxDepth):
    for depth in 0..maxDepth:
        parent = map()
        parent[start] = null
        found = DLS(start, depth, parent)
        if found != failure:
            return RECONSTRUCT_PATH(parent, found)
    return failure

function DLS(s, limit, parent):
    if GOAL_TEST(s): return s
    if limit == 0: return failure
    for a in ACTIONS(s):
        s2 = RESULT(s,a)
        if s2 not in parent:              # avoid cycles on current path
            parent[s2] = (s, a)
            found = DLS(s2, limit-1, parent)
            if found != failure: return found
    return failure

function RECONSTRUCT_PATH(parent, goal):
    path = []
    cur = goal
    while cur != null:
        prepend cur to path
        cur = parent[cur].prevState if parent[cur] != null else null
    return path

3) A* with admissible heuristic hook

Q: Implement A_STAR(start, h) for maze navigation where h(s) is a heuristic.

function A_STAR(start, h):
    open = minPriorityQueue()          # key = f = g + h
    g = map(default=+infinity)
    parent = map()
    g[start] = 0
    parent[start] = null
    open.push(start, g[start] + h(start))

    while not open.isEmpty():
        s = open.popMin()
        if GOAL_TEST(s):
            return RECONSTRUCT_PATH(parent, s)

        for a in ACTIONS(s):
            s2 = RESULT(s,a)
            tentative = g[s] + COST(s,a,s2)     # usually 1 in grids
            if tentative < g[s2]:
                g[s2] = tentative
                parent[s2] = (s, a)
                open.pushOrDecreaseKey(s2, g[s2] + h(s2))
    return failure

4) Depth cutoff search (DFS with cutoff)

Q: Write DFS_CUTOFF(start, cutoff) that returns failure if not found within depth.

function DFS_CUTOFF(start, cutoff):
    parent = map()
    parent[start] = null
    found = DLS(start, cutoff, parent)     # reuse DLS from IDDFS
    if found == failure: return failure
    return RECONSTRUCT_PATH(parent, found)

5) Multi-agent “boids” step update

Q: Implement one simulation tick for boids with separation, alignment, cohesion.

function STEP_BOIDS(boids, params):
    # boid: (pos, vel)
    newBoids = []
    for i in 0..len(boids)-1:
        b = boids[i]
        neighbors = NEIGHBORS(boids, i, params.radius)

        sep = vector(0,0)
        ali = vector(0,0)
        coh = vector(0,0)

        for j in neighbors:
            diff = b.pos - boids[j].pos
            sep += normalize(diff) / (epsilon + norm(diff))
            ali += boids[j].vel
            coh += boids[j].pos

        if len(neighbors) > 0:
            ali = (ali / len(neighbors)) - b.vel
            coh = (coh / len(neighbors)) - b.pos

        accel = params.wSep*sep + params.wAli*ali + params.wCoh*coh
        v2 = limit(b.vel + accel, params.maxSpeed)
        p2 = b.pos + v2
        newBoids.append( (p2, v2) )
    return newBoids

Chapter 2 — Local Search and Optimization
1) Objective function for 8-queens

Q: Write CONFLICTS(board) where board[col]=row.

function CONFLICTS(board):
    n = len(board)
    c = 0
    for i in 0..n-1:
        for j in i+1..n-1:
            sameRow = (board[i] == board[j])
            sameDiag = (abs(board[i]-board[j]) == abs(i-j))
            if sameRow or sameDiag:
                c += 1
    return c

2) Hill climbing with “best neighbor”

Q: Implement hill climbing to minimize CONFLICTS.

function HILL_CLIMB(board):
    while true:
        cur = CONFLICTS(board)
        bestBoard = board
        best = cur

        for col in 0..n-1:
            originalRow = board[col]
            for row in 0..n-1:
                if row == originalRow: continue
                candidate = copy(board)
                candidate[col] = row
                score = CONFLICTS(candidate)
                if score < best:
                    best = score
                    bestBoard = candidate

        if best == cur:          # no improvement => stuck
            return board
        board = bestBoard

3) Simulated annealing

Q: Implement simulated annealing for 8-queens with random neighbor.

function SIM_ANNEAL(board, T0, alpha, steps):
    T = T0
    for t in 1..steps:
        if CONFLICTS(board) == 0: return board

        next = RANDOM_NEIGHBOR(board)
        delta = CONFLICTS(next) - CONFLICTS(board)   # minimize
        if delta <= 0:
            board = next
        else:
            p = exp(-delta / T)
            if random01() < p:
                board = next

        T = alpha * T
        if T < 1e-6: break
    return board

function RANDOM_NEIGHBOR(board):
    n = len(board)
    col = randomInt(0,n-1)
    row = randomInt(0,n-1)
    nb = copy(board)
    nb[col] = row
    return nb

4) Genetic algorithm for 8-queens

Q: Implement a GA loop with selection, crossover, mutation.

function GA_8QUEENS(popSize, generations, mutRate):
    pop = [RANDOM_BOARD() for k in 1..popSize]

    for g in 1..generations:
        if any(CONFLICTS(b)==0 for b in pop): return argmin(pop, CONFLICTS)

        # fitness higher is better; use inverse conflicts
        fitness = [1 / (1 + CONFLICTS(b)) for b in pop]

        newPop = []
        while len(newPop) < popSize:
            p1 = TOURNAMENT_SELECT(pop, fitness, k=3)
            p2 = TOURNAMENT_SELECT(pop, fitness, k=3)
            (c1,c2) = ONE_POINT_CROSSOVER(p1,p2)
            c1 = MUTATE(c1, mutRate)
            c2 = MUTATE(c2, mutRate)
            newPop.append(c1)
            if len(newPop) < popSize: newPop.append(c2)

        pop = newPop
    return argmin(pop, CONFLICTS)

function ONE_POINT_CROSSOVER(a,b):
    n = len(a)
    cut = randomInt(1, n-2)
    c1 = a[0:cut] + b[cut:n]
    c2 = b[0:cut] + a[cut:n]
    return (c1,c2)

function MUTATE(board, mutRate):
    if random01() < mutRate:
        col = randomInt(0, len(board)-1)
        board[col] = randomInt(0, len(board)-1)
    return board

5) Continuous local search (airport-style)

Q: Minimize sum of squared distances from candidate airport (x,y) to towns.

function AIRPORT_OBJECTIVE(x, y, towns):
    # towns: list of (tx,ty, weight)
    s = 0
    for (tx,ty,w) in towns:
        dx = x - tx; dy = y - ty
        s += w * (dx*dx + dy*dy)
    return s

function RANDOM_SEARCH_CONTINUOUS(towns, bounds, iters):
    best = null
    bestVal = +infinity
    for i in 1..iters:
        x = randomUniform(bounds.xMin, bounds.xMax)
        y = randomUniform(bounds.yMin, bounds.yMax)
        v = AIRPORT_OBJECTIVE(x,y,towns)
        if v < bestVal:
            bestVal = v
            best = (x,y)
    return best

Chapter 3 — Adversarial Search and Games
1) Minimax for tic-tac-toe

Q: Implement MINIMAX(s, player) returning best move and value.

function MINIMAX_DECISION(state, player):
    bestMove = null
    bestVal = -infinity
    for a in ACTIONS(state):
        v = MIN_VALUE(RESULT(state,a), player)
        if v > bestVal:
            bestVal = v
            bestMove = a
    return (bestMove, bestVal)

function MAX_VALUE(s, player):
    if TERMINAL(s): return UTILITY(s, player)
    v = -infinity
    for a in ACTIONS(s):
        v = max(v, MIN_VALUE(RESULT(s,a), player))
    return v

function MIN_VALUE(s, player):
    if TERMINAL(s): return UTILITY(s, player)
    v = +infinity
    for a in ACTIONS(s):
        v = min(v, MAX_VALUE(RESULT(s,a), player))
    return v

2) Alpha–beta pruning

Q: Add alpha–beta pruning to minimax.

function ALPHABETA_DECISION(state, player):
    bestMove = null
    alpha = -infinity
    beta = +infinity
    bestVal = -infinity

    for a in ACTIONS(state):
        v = AB_MIN(RESULT(state,a), player, alpha, beta)
        if v > bestVal:
            bestVal = v
            bestMove = a
        alpha = max(alpha, bestVal)
    return (bestMove, bestVal)

function AB_MAX(s, player, alpha, beta):
    if TERMINAL(s): return UTILITY(s, player)
    v = -infinity
    for a in ACTIONS(s):
        v = max(v, AB_MIN(RESULT(s,a), player, alpha, beta))
        if v >= beta: return v
        alpha = max(alpha, v)
    return v

function AB_MIN(s, player, alpha, beta):
    if TERMINAL(s): return UTILITY(s, player)
    v = +infinity
    for a in ACTIONS(s):
        v = min(v, AB_MAX(RESULT(s,a), player, alpha, beta))
        if v <= alpha: return v
        beta = min(beta, v)
    return v

3) Expectiminimax (stochastic)

Q: Implement expectiminimax with CHANCE nodes.

function EXPECTIMINIMAX(s, player):
    if TERMINAL(s): return UTILITY(s, player)
    if NODE_TYPE(s) == "MAX":
        v = -infinity
        for a in ACTIONS(s):
            v = max(v, EXPECTIMINIMAX(RESULT(s,a), player))
        return v
    if NODE_TYPE(s) == "MIN":
        v = +infinity
        for a in ACTIONS(s):
            v = min(v, EXPECTIMINIMAX(RESULT(s,a), player))
        return v
    if NODE_TYPE(s) == "CHANCE":
        v = 0
        for (outcome, p) in OUTCOMES(s):    # each outcome is a successor state
            v += p * EXPECTIMINIMAX(outcome, player)
        return v

4) Monte Carlo rollouts for move choice

Q: Choose move by random playouts (lightweight MCTS-style).

function MONTE_CARLO_MOVE(state, player, rolloutsPerMove):
    bestMove = null
    bestScore = -infinity

    for a in ACTIONS(state):
        total = 0
        for k in 1..rolloutsPerMove:
            s = RESULT(state, a)
            total += RANDOM_PLAYOUT(s, player)
        avg = total / rolloutsPerMove
        if avg > bestScore:
            bestScore = avg
            bestMove = a

    return bestMove

function RANDOM_PLAYOUT(s, player):
    while not TERMINAL(s):
        a = randomChoice(ACTIONS(s))
        s = RESULT(s, a)
    return UTILITY(s, player)

5) Compute Nash equilibrium for 2×2 (pure strategy check)

Q: Given payoff matrices A (row player) and B (col player), find pure Nash equilibria.

function PURE_NASH_2x2(A, B):
    equilibria = []
    for i in 0..1:          # row action
        for j in 0..1:      # col action
            # is (i,j) best response for row?
            rowBest = true
            for i2 in 0..1:
                if A[i2][j] > A[i][j]:
                    rowBest = false
            # is (i,j) best response for col?
            colBest = true
            for j2 in 0..1:
                if B[i][j2] > B[i][j]:
                    colBest = false
            if rowBest and colBest:
                add (i,j) to equilibria
    return equilibria

Chapter 4 — CSPs
1) Sudoku/map coloring CSP representation

Q: Write a CSP data structure and initialize map-coloring.

function MAKE_MAP_COLORING_CSP(regions, edges, colors):
    csp = {}
    csp.vars = regions
    csp.dom = map()
    for r in regions: csp.dom[r] = copy(colors)
    csp.neigh = map(default=[])
    for (u,v) in edges:
        csp.neigh[u].append(v)
        csp.neigh[v].append(u)
    return csp

2) Backtracking search (search + inference hook)

Q: Implement BACKTRACK(assignment, csp) with forward checking.

function BACKTRACK(assignment, csp):
    if isComplete(assignment, csp.vars): return assignment
    X = SELECT_UNASSIGNED_VAR(assignment, csp)      # e.g., MRV
    for v in ORDER_VALUES(X, assignment, csp):
        if CONSISTENT(X, v, assignment, csp):
            assignment[X] = v
            savedDomains = copyDomains(csp.dom)
            if FORWARD_CHECK(X, v, assignment, csp):
                result = BACKTRACK(assignment, csp)
                if result != failure: return result
            csp.dom = savedDomains
            remove X from assignment
    return failure

function FORWARD_CHECK(X, v, assignment, csp):
    for Y in csp.neigh[X]:
        if Y not in assignment:
            remove v from csp.dom[Y]        # constraint Y != X for map coloring
            if csp.dom[Y] is empty: return false
    return true

3) AC-3 arc consistency

Q: Implement AC-3.

function AC3(csp):
    Q = queue()
    for X in csp.vars:
        for Y in csp.neigh[X]:
            Q.enqueue((X,Y))

    while not Q.isEmpty():
        (X,Y) = Q.dequeue()
        if REVISE(csp, X, Y):
            if csp.dom[X] is empty: return false
            for Z in csp.neigh[X]:
                if Z != Y:
                    Q.enqueue((Z,X))
    return true

function REVISE(csp, X, Y):
    revised = false
    for x in copy(csp.dom[X]):
        if not exists y in csp.dom[Y] such that CONSTRAINT_OK(X,x,Y,y):
            remove x from csp.dom[X]
            revised = true
    return revised

function CONSTRAINT_OK(X,x,Y,y):
    return x != y      # map coloring; replace for Sudoku etc.

4) Tree-structured CSP solver (DP on tree)

Q: Solve tree-structured CSP given root.

function TREE_CSP_SOLVE(csp, root):
    order = TOPOLOGICAL_ORDER_FROM_ROOT(csp, root)      # parents before children
    parent = PARENTS_FROM_ROOT(csp, root)

    # make arc-consistent from leaves upward
    for X in reverse(order):
        for Y in csp.neigh[X]:
            if parent[X] == Y:
                REVISE(csp, Y, X)

    assignment = {}
    assignment[root] = any value in csp.dom[root]
    for X in order:
        if X == root: continue
        P = parent[X]
        assignment[X] = any v in csp.dom[X] with CONSTRAINT_OK(P, assignment[P], X, v)
    return assignment

5) Cutset conditioning (high-level)

Q: Implement cutset conditioning: choose cutset variables, enumerate assignments, solve remaining as tree CSP.

function CUTSET_CONDITIONING(csp, cutsetVars):
    for cutAssign in ALL_ASSIGNMENTS(cutsetVars, csp.dom):
        csp2 = COPY_CSP(csp)
        if not APPLY_ASSIGNMENT(csp2, cutAssign): continue
        if not AC3(csp2): continue

        # assume removing cutset makes graph a tree; pick a root among remaining
        root = any var not in cutsetVars
        sol = TREE_CSP_SOLVE(csp2, root)
        if sol != failure:
            return MERGE(sol, cutAssign)
    return failure

Chapter 5 — Logic
1) Unification

Q: Implement UNIFY(x, y, subst) for terms/vars.

function UNIFY(x, y, subst):
    x = APPLY(subst, x)
    y = APPLY(subst, y)
    if x == y: return subst
    if isVar(x): return UNIFY_VAR(x, y, subst)
    if isVar(y): return UNIFY_VAR(y, x, subst)
    if isFunc(x) and isFunc(y) and x.name==y.name and len(x.args)==len(y.args):
        for k in 0..len(x.args)-1:
            subst = UNIFY(x.args[k], y.args[k], subst)
            if subst == failure: return failure
        return subst
    return failure

function UNIFY_VAR(v, x, subst):
    if v in subst: return UNIFY(subst[v], x, subst)
    if occursCheck(v, x, subst): return failure
    subst[v] = x
    return subst

2) Forward chaining (Horn clauses)

Q: Implement forward chaining to derive query.

function FORWARD_CHAIN(rules, facts, query):
    agenda = queue(facts)
    inferred = set()
    count = map(rule -> number of unsatisfied premises)

    for r in rules:
        count[r] = len(r.premises)

    while not agenda.isEmpty():
        p = agenda.dequeue()
        if p == query: return true
        if p not in inferred:
            inferred.add(p)
            for r in rules where p in r.premises:
                count[r] -= 1
                if count[r] == 0:
                    agenda.enqueue(r.conclusion)
    return false

3) Backward chaining (goal-driven)

Q: Implement backward chaining proof search for a query.

function BACKWARD_CHAIN(rules, facts, goal, visited):
    if goal in facts: return true
    if goal in visited: return false
    visited.add(goal)

    for r in rules where r.conclusion == goal:
        ok = true
        for prem in r.premises:
            if not BACKWARD_CHAIN(rules, facts, prem, visited):
                ok = false
                break
        if ok: return true
    return false

4) Resolution (propositional)

Q: Implement a resolution loop to prove KB ⊨ q by refutation.

function RESOLUTION_ENTAILS(KB_clauses, q):
    clauses = set(KB_clauses ∪ CNF(¬q))
    new = set()

    while true:
        pairs = ALL_PAIRS(clauses)
        for (c1,c2) in pairs:
            resolvents = RESOLVE(c1,c2)
            if {} in resolvents: return true      # empty clause
            new = new ∪ resolvents
        if new ⊆ clauses: return false
        clauses = clauses ∪ new

5) Inference as search (generic)

Q: Implement proof search as BFS over “derived facts”.

function PROOF_BFS(rules, facts, query):
    frontier = queue()
    visitedFacts = set(facts)
    frontier.enqueue(facts)

    while not frontier.isEmpty():
        currentFacts = frontier.dequeue()
        if query in currentFacts: return true

        for r in rules:
            if all prem in currentFacts:
                nextFacts = currentFacts ∪ {r.conclusion}
                if r.conclusion not in visitedFacts:
                    visitedFacts.add(r.conclusion)
                    frontier.enqueue(nextFacts)
    return false

Chapter 6 — Probabilistic Reasoning + Learning
1) Exact inference in Bayes net (enumeration)

Q: Implement ENUM_ASK(X, e, bn).

function ENUM_ASK(X, e, bn):
    Q = map(value -> 0)
    for x in DOMAIN(X):
        e2 = copy(e); e2[X] = x
        Q[x] = ENUM_ALL(bn.vars, e2, bn)
    return NORMALIZE(Q)

function ENUM_ALL(vars, e, bn):
    if vars is empty: return 1
    Y = first(vars)
    rest = vars[1:]
    if Y in e:
        return P(Y=e[Y] | parents(Y)=e[parents(Y)], bn) * ENUM_ALL(rest, e, bn)
    else:
        s = 0
        for y in DOMAIN(Y):
            e2 = copy(e); e2[Y] = y
            s += P(Y=y | parents(Y)=e2[parents(Y)], bn) * ENUM_ALL(rest, e2, bn)
        return s

2) Likelihood weighting

Q: Approximate P(X|e) using likelihood weighting.

function LIKELIHOOD_WEIGHTING(X, e, bn, N):
    W = map(value -> 0)
    for i in 1..N:
        (sample, w) = WEIGHTED_SAMPLE(bn, e)
        W[sample[X]] += w
    return NORMALIZE(W)

function WEIGHTED_SAMPLE(bn, e):
    w = 1
    sample = copy(e)
    for Y in bn.topoOrder:
        if Y in e:
            w *= P(Y=e[Y] | parents(Y)=sample[parents(Y)], bn)
        else:
            sample[Y] = SAMPLE_FROM(P(Y | parents(Y)=sample[parents(Y)], bn))
    return (sample, w)

3) HMM forward algorithm (filtering)

Q: Implement HMM filtering BELIEF_t = α * O_t * T^T * BELIEF_{t-1}.

function HMM_FILTER(priorBelief, observations, T, O):
    # belief is vector over hidden states
    belief = priorBelief
    for obs in observations:
        belief = TRANSPOSE(T) * belief
        belief = O[obs] * belief          # O[obs] is diagonal matrix or elementwise mult
        belief = NORMALIZE_VECTOR(belief)
    return belief

4) Particle filtering

Q: Implement particle filter for DBN/HMM.

function PARTICLE_FILTER(particles, observation, TRANSITION, OBS_LIK, N):
    # 1) propagate
    for i in 1..N:
        particles[i] = TRANSITION(particles[i])

    # 2) weight by observation likelihood
    weights = []
    for i in 1..N:
        weights[i] = OBS_LIK(observation, particles[i])

    # 3) resample
    particles = RESAMPLE(particles, weights, N)
    return particles

5) Supervised learning: train/test split + accuracy

Q: Implement a generic supervised training pipeline.

function TRAIN_EVAL(dataset, model, lossFn, optimizer, epochs):
    (trainSet, testSet) = SPLIT(dataset, ratio=0.8, shuffle=true)

    for e in 1..epochs:
        for (x,y) in trainSet:
            yhat = model.forward(x)
            L = lossFn(yhat, y)
            grads = model.backward(L)
            optimizer.step(model.params, grads)

    correct = 0
    total = 0
    for (x,y) in testSet:
        yhat = argmax(model.forward(x))
        if yhat == y: correct += 1
        total += 1
    return correct / total

Chapter 7 — Neural Nets + Transformers + “Make your own ChatGPT”
1) Feedforward + backprop (2-layer MLP)

Q: Write one training step for a 2-layer net with ReLU + softmax.

function TRAIN_STEP(x, y, W1, b1, W2, b2, lr):
    # forward
    z1 = W1*x + b1
    h  = RELU(z1)
    z2 = W2*h + b2
    p  = SOFTMAX(z2)

    # loss (cross-entropy)
    L = -log(p[y])

    # backward
    dz2 = p
    dz2[y] -= 1                     # softmax+CE gradient
    dW2 = dz2 * transpose(h)
    db2 = dz2

    dh  = transpose(W2) * dz2
    dz1 = dh * RELU_DERIV(z1)
    dW1 = dz1 * transpose(x)
    db1 = dz1

    # update
    W1 -= lr*dW1; b1 -= lr*db1
    W2 -= lr*dW2; b2 -= lr*db2
    return (W1,b1,W2,b2,L)

2) Vision example: single convolution forward

Q: Implement a single 2D convolution (no padding, stride 1).

function CONV2D(image, kernel):
    H = image.height; W = image.width
    kH = kernel.height; kW = kernel.width
    outH = H - kH + 1
    outW = W - kW + 1
    out = matrix(outH, outW, 0)

    for i in 0..outH-1:
        for j in 0..outW-1:
            s = 0
            for a in 0..kH-1:
                for b in 0..kW-1:
                    s += image[i+a][j+b] * kernel[a][b]
            out[i][j] = s
    return out

3) Self-attention (single head)

Q: Implement single-head scaled dot-product attention.

function ATTENTION(Q, K, V):
    # Q,K,V: matrices (seqLen x d)
    scores = (Q * transpose(K)) / sqrt(d)
    weights = ROWWISE_SOFTMAX(scores)     # each row sums to 1
    return weights * V

4) Transformer block (high-level)

Q: Implement a transformer block forward pass with residual + layernorm.

function TRANSFORMER_BLOCK(X, params):
    # X: (seqLen x dModel)
    A = ATTENTION(X*params.Wq, X*params.Wk, X*params.Wv) * params.Wo
    X1 = LAYERNORM(X + A)

    F = GELU(X1*params.W1 + params.b1) * params.W2 + params.b2
    Y = LAYERNORM(X1 + F)
    return Y

5) “Make your own ChatGPT”: tiny LM training loop

Q: Write pseudocode to train a next-token transformer LM.

function TRAIN_LM(corpusText, tokenizer, model, optimizer, steps, seqLen, batchSize):
    tokens = tokenizer.encode(corpusText)

    for step in 1..steps:
        batchX = []
        batchY = []
        for b in 1..batchSize:
            i = randomInt(0, len(tokens)-seqLen-2)
            x = tokens[i : i+seqLen]              # input tokens
            y = tokens[i+1 : i+seqLen+1]          # next-token targets
            batchX.append(x)
            batchY.append(y)

        logits = model.forward(batchX)            # (B, seqLen, vocab)
        L = CROSS_ENTROPY(logits, batchY)         # token-wise
        grads = model.backward(L)
        optimizer.step(model.params, grads)

    return model