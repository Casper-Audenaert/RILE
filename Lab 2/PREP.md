# Lab 2: Presentation prep

## 0. Explained simply

### What's in this folder

| File | What it is |
|---|---|
| `Reinforcement_Learning___Lab_2-2.pdf` | The handout: the tasks and questions. |
| `RL_LAB_2.ipynb` | The notebook: the grid, the two fixed policies, the simulator and value iteration. |
| `SOLUTION.md` | What we changed in the code, the results, and why it works. |
| `PREP.md` | This file: what to say and what you might be asked. |

### The concepts in everyday words

- **MDP (Markov decision process).** Think of a board game. The squares are **states** (which router has the packet), the moves you can choose are **actions** (send up/down/left/right), a die decides where you actually land (**transitions**), and you lose a point for every move (**reward −1**). "Markov" means only the square you are on matters, not how you got there.
- **Unreliable links.** The die is unfair and different on every square: usually the packet goes where you sent it, sometimes it slips sideways, and sometimes it just stays (the transmission failed).
- **Policy.** A rulebook: "at this router, always send in this direction". Always-Right and Always-Down are two very simple rulebooks.
- **Value V(s).** "Starting here and following the rulebook, how many points do I lose on average?" Here that is minus the expected number of hops. V = −10 means "about 10 hops". Closer to 0 is better.
- **Policy evaluation with a linear system.** Each router's value = −1 (this hop) + the average value of where the packet lands next. Write that down for every router and you get a set of equations; solving them all at once gives the exact values.
- **Proper policy.** A rulebook that always gets the packet home eventually. Always-Up is not: once the packet is in the top row, pushing up does nothing, and slips only move it left or right along that row, so it never arrives.
- **Why Always-Right is so bad (≈ −196).** The packet reaches the right wall quickly. Then "send right" just bounces off the wall, and the packet only moves down (or up!) by accident through slips. It is like trying to walk down a corridor by bumping into the side wall.
- **Monte Carlo.** Forget the equations: just play the game many times (2000 episodes) and average the score. It needs no knowledge of the die, only the outcomes. That is what makes it reinforcement learning.
- **Value iteration.** Start by guessing 0 everywhere. Then repeat: every router asks "given my neighbours' current estimates, which direction is best, and what does that give me?" The knowledge spreads out from the goal like ripples on water, one hop per sweep. When nothing changes any more (47 sweeps here), you have the optimal values.
- **Optimal policy.** At each router, pick the direction that gives the best value one step ahead. The result uses only Right and Down and avoids the bad links, bringing (0,0) from about 200 hops down to about 10.
- **γ (discount).** How much future costs count. γ = 1 means every hop costs the same, now or later.

### The whole lab in one paragraph

A packet has to cross a grid of flaky routers. We first measured how good two simple rulebooks are, once exactly with equations and once by simulating many deliveries. Both are terrible (about 200 hops) because they get stuck against a wall. Value iteration then computes the best direction at every router from the known link probabilities, and the packet arrives in about 10 hops.

### The notebook, cell by cell

| # | Cell | What it does, in one line |
|---|---|---|
| 1 | "Run this first" (text) | Reminder to run the setup cell. |
| 2 | Setup (big code cell) | Builds everything: `GridSSP` (the 5×5 router grid with its own slip/stay odds per router, exact policy evaluation, simulation, value iteration), `build_policy`, the heatmap plot, `run_many_simulations` and the animation. Nothing to change. |
| 3 | Task 1 (text) | Asks to describe the MDP. |
| 4 | Task 1 code | Prints and plots where the packet can land from one router with one action. Default (2,2) 'U': 0.62 up, 0.10 / 0.05 sideways, 0.23 stays. Change `state` and `action` to explore. |
| 5 | Task 2 (text) | Asks to evaluate two fixed policies. |
| 6 | Task 2 code | Always-Right: solves the equations for the exact values and shows a heatmap. Good only on the bottom row. |
| 7 | Task 2 (Part 2) | Same for Always-Down. Good only in the right column. (The variable is called `V_up` but holds the Always-Down values.) |
| 8 | Task 3 (text) | Asks to simulate and estimate values by averaging. |
| 9 | Task 3 Part 1 | Plays one episode of Always-Right from (0,0) and prints the path. You see it stuck against the right wall. |
| 10 | Task 3 Part 2 | One episode of a random policy (70% R, 30% D) from (2,2). |
| 11 | Task 3 Part 3 (optional) | Animates one episode on the grid. |
| 12 | Task 3 Part 4 | Monte Carlo: plays many episodes and averages the score. Run with `pi_right`, then `pi_down`, and raise `n_sims` to 2000 to land close to the exact values from Task 2. |
| 13 | Task 4 (text) | Asks to explain "optimal" and value iteration. |
| 14 | Task 4 code | Runs value iteration (47 sweeps) and shows the optimal heatmap: about −10 from (0,0). |
| 15 | Empty | Use it to print the optimal policy (see §2). |

## 1. Opening (30 seconds)

> "We route a packet across a 5×5 wireless grid to router (4,4). Links are unreliable: the packet can slip sideways or stay where it is. Every transmission costs −1, so we want to minimise the expected number of hops. This is a stochastic shortest-path MDP. We evaluated two fixed policies exactly (linear system) and by simulation (Monte Carlo), then found the optimal policy with value iteration."

## 2. Demo flow

| Step | Show | Say |
|---|---|---|
| 1 | Task 1 bar plot, (2,2) 'U' | "62% it goes up as intended, 10% and 5% it slips sideways, 23% it stays (failed transmission). Each router has its own numbers." Then try (0,3) 'R'. |
| 2 | Task 2 heatmaps | "V(s) = −expected number of hops. Always-R is only good on the bottom row, and always-D only in the right column. Both are about −200 from (0,0)." |
| 3 | Task 3 Part 1/2 trajectories | "Look at always-R: it reaches (0,4) quickly and then sits against the right wall for ~100 steps." |
| 4 | Task 3 Part 4 | Run with `pi_right`, then change it to `pi_down`. Raise `n_sims` to 2000. "The MC average approaches the linear-system value." |
| 5 | Task 4 heatmap | "V\*(0,0) ≈ −10, about 20 times better than the fixed policies. Values decrease smoothly by about 1.3 per hop toward the goal." |

To show the optimal policy, type in a new cell:
```python
for i in range(5): print(' '.join(pi_opt[(i, j)] for j in range(5)))
```

## 3. Formulas to know by heart

- Policy evaluation: V^π = R^π + P^π V^π  ⇒  (I − P_T) v_T = R_T, with V(goal) = 0
- Iterative evaluation: V_{k+1}(s) = Σ_{s'} P(s'|s,π(s)) [R + γV_k(s')]
- Value iteration: V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV_k(s')]
- Greedy policy: π\*(s) = argmax_a Σ_{s'} P(s'|s,a) [R + γV\*(s')]
- Monte Carlo: V̂(s₀) = (1/n) Σ_i G_i

## 4. Numbers to remember

- (2,2) with 'U': intended 0.62, slips 0.10 and 0.05, stay 0.23.
- V(0,0): always-R ≈ −196, always-D ≈ −203, optimal ≈ −10.2.
- MC (2000 episodes): always-R ≈ −194 ± 8, always-D ≈ −196 ± 8, success rate 100%.
- Value iteration converged in 47 iterations (tol 1e-8).
- Optimal policy: only R and D moves.

## 5. Likely questions and answers

**What are the states, actions, transitions and rewards?**
- **States:** the router holding the packet, 25 in total. (4,4) is absorbing.
- **Actions:** U/D/L/R, meaning "try to forward in that direction".
- **Transitions:** intended direction with probability 0.5–0.8, slip to each perpendicular side with 0.05–0.10, otherwise stay. Moves off the grid are clipped (the packet stays). The probabilities are different for every (s,a).
- **Reward:** −1 per step, 0 at the goal.

**Is it Markov?**
Yes. Where the packet goes next depends only on the current router and the action, not on how it got there.

**What does a high stay probability mean?**
The transmission failed: congestion, interference or packet loss. It costs a step and makes no progress. Good routing avoids such routers.

**Why is this called a stochastic shortest path problem?**
It is undiscounted (γ = 1), there is a terminal goal, and every step has a positive cost. Minimising cost means minimising the expected number of hops.

**What does V^π(2,2) = −x mean?**
Starting at (2,2) and following π, it takes on average x transmissions to deliver the packet.

**Why do we need V(goal) = 0 with γ = 1?**
With γ = 1, every row of P sums to 1, so (I − P) is singular: if V solves the equation, so does V + constant. Fixing the goal's value to 0 pins down the constant. We then solve only on the transient states, where (I − P_T) is invertible *if the policy is proper*.

**What is a proper policy? Give an improper one.**
A policy is proper if it reaches the goal with probability 1 from every state.
- **Always U** is improper. Once the packet is in row 0, 'U' keeps it there, and slips only move it left or right along row 0, so it can never get down to (4,4). The expected cost is −∞ and the linear system is singular.
- **Always L** is improper for the same reason.

**What changes if γ < 1?**
(I − γP) is always invertible, so no boundary condition is needed. Values become bounded even for improper policies, and future costs are discounted. The meaning changes: V is no longer exactly minus the number of hops.

**Why are always-R and always-D so bad (≈ −200)?**
Always-R reaches the right wall quickly. There, 'R' is clipped, so the packet only moves through slips: down with probability ≈ 0.05–0.1, but *up* with similar probability. This is a slow random walk along the wall. Always-D has the same problem on the bottom row.

**Which fixed policy is better?**
They are similar. Always-R is slightly better from (0,0) (−196 vs. −203). Each is good only in the region where its direction leads straight to the goal: the last row for R, the last column for D.

**Linear solve vs. iterative DP: what's the difference?**
- The direct solve is exact and costs O(n³), which is trivial for 24 states.
- Iterative evaluation repeats the Bellman equation until convergence. It is cheaper per step and scales to large problems, but it is approximate (stopped at a tolerance).
- Both give the same fixed point for a proper policy.

**What is a trajectory, and why does it differ between episodes?**
A trajectory is the sequence s₀, a₀, r₁, s₁, … of one episode. It differs because the transitions are sampled from P(s'|s,a), and a stochastic policy (e.g. 70% R, 30% D) samples the actions too. The RNG seed controls the randomness: the same seed gives the same trajectory.

**How does the MC average estimate V^π(s₀)?**
V^π(s₀) = E[G | s₀], and each episode's return G is an independent unbiased sample of it. By the law of large numbers the mean converges, with standard error σ/√n. MC never uses P: it only needs observed rewards. That makes it **model-free**, a reinforcement learning method.

**Why does the MC value differ from the linear-system value?**
Two reasons:
1. **Noise.** With 100 episodes the always-R estimate was −226. Returns vary a lot because some episodes get stuck against the wall for hundreds of steps. With 2000 episodes it was −194 ± 8.
2. **Convention.** `evaluate_policy` charges −1 for the final step into the goal; the simulator charges 0 for it. So MC is systematically about 1 higher.

**What does "optimal" mean?**
π\* gives the best expected value in *every* state at the same time: V\*(s) = max_π V^π(s). There is no state where another policy does better.

**How does value iteration work, and why does it converge?**
- Start from V = 0.
- Apply the Bellman optimality backup to every non-goal state, keeping V(goal) = 0. Each sweep spreads "distance to goal" information one hop further.
- Stop when the largest change is below a tolerance.
- In an SSP where proper policies exist and improper ones have infinite cost, the backup converges to the unique V\*.
- The code is synchronous: it uses `V_old` for the whole sweep.

**How do you get π\* from V\*?**
One-step lookahead: for each action, compute the expected immediate reward plus the value of the next state, then take the argmax. This requires the model P, which is why value iteration is model-based.

**Why does π\* use only R and D?**
Moving away from the goal costs at least one extra step and gives nothing back. π\* chooses between R and D at each router based on that router's link quality, and it never pushes into a wall.

**Value iteration vs. policy iteration?**
Policy iteration alternates exact evaluation (Task 2) with greedy improvement. It needs fewer iterations, but each one is more expensive. Value iteration does one optimality backup per sweep.

**Is value iteration reinforcement learning?**
No. It is dynamic programming: it needs the full model P and R. Monte Carlo (Task 3) is the RL part, because it learns from samples only.

## 6. Traps to avoid

- The handout text says "always Up and always Down" for Task 2, but the code (and policy 1) is **always Right**. Always Up is improper.
- In the notebook the always-down value is stored in a variable named `V_up`. It is the always-down value, despite the name.
- Rewards are negative, so "higher" (closer to 0) is better.
