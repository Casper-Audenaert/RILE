# Lab 2: MDP Modeling and Dynamic Programming (solution notes)

## What was changed in the notebook

Nothing. The environment, policy evaluation, value iteration and simulation are all provided, so the notebook only has to be run.

The Task 3 main experiment needs one parameter change in the **Task 3 (Part 4)** cell:
- Run it with `pi_right` (the default), then again with `pi_down = {s: 'D' for s in env.states}`.
- Raise `n_sims`, for example to 2000, for a tighter estimate.

The numbers below are from that change with n_sims = 2000. The saved notebook output shows the default run: always R with n_sims = 100, mean return −226.4. That estimate is noisy because 100 episodes is few.

## Task 1: The MDP

- **States**: the router holding the packet, (i, j) on a 5×5 grid. (4,4) is the absorbing goal.
- **Actions**: U, D, L, R, meaning "try to forward to that neighbour".
- **Transitions P(s'|s,a)**: each (s,a) has its own probabilities. The packet goes in the intended direction with probability 0.5–0.8, slips sideways (perpendicular) with probability 0.05–0.10 each way, and *stays* with the remaining probability. A move off the grid is clipped, so the packet stays where it is.
- **Reward**: −1 per transmission. The goal is absorbing, so the objective is to minimise the expected number of hops (a stochastic shortest-path problem).

For (2,2) with 'U': (1,2) with p = 0.62, (2,1) with 0.10, (2,3) with 0.05, and stay (2,2) with 0.23. The intended neighbour is the most likely next state. A high *stay* probability means the link failed or the packet had to be retransmitted, which is congestion or interference at that router. Slips are packets that a different neighbour picked up. Routers with a high stay probability are "slow", and a good policy routes around them.

## Task 2: Policy evaluation via a linear system

**Meaning.** V^π(s) is the expected total reward from s when following π. Here that is minus the expected number of transmissions until delivery. For example, V^R(2,2) is how many hops "always right" needs on average from (2,2), with a minus sign.

**Method.** For a fixed π the Bellman equation is linear: V = R^π + P^π V. With γ = 1, the matrix (I − P) is singular, because the goal row of P is a unit vector, so V + const is also a solution. Fixing V(goal) = 0 and keeping only the transient states gives (I − P_T) v = R_T. For a *proper* policy, the one that reaches the goal with probability 1, P_T has spectral radius < 1, so (I − P_T) is invertible and the solution is unique. With γ < 1, I − γP is always invertible, so no boundary condition is needed and even improper policies get finite values.

**Comparison.** Both fixed policies are poor. Always R reaches column 4 and then gets "stuck" pushing right into the wall. From there it only moves down when it slips, and with probability 0.05–0.10 per step that takes many steps. Always D has the same problem in row 4. Both have V(0,0) of about −200. States on the last row (for R) or the last column (for D) are good, and everything else is bad. Neither policy is clearly better. Always R is slightly better from (0,0) (−196 against −203).

**DP alternative.** Repeat V_{k+1}(s) ← Σ P(s'|s,π(s)) [R(s,π(s),s') + V_k(s')] with V(goal) = 0 until max|V_{k+1} − V_k| < tol. This is a contraction on the transient states for a proper policy, so it converges to the same fixed point as the direct solve.

**Note on the reward convention.**
- `evaluate_policy` charges −1 for *every* step, including the final one into the goal.
- `env.reward`, `simulate` and `value_iteration` charge 0 for the step that enters the goal.
- Every episode ends with exactly one such step, so the two conventions differ by exactly 1 in every non-goal state. They describe the same values with a shift. I checked this numerically; it is not part of the notebook.

## Task 3: Monte Carlo policy evaluation

| Policy (from (0,0)) | MC mean return (95% CI) | Mean steps | Success | Linear V(0,0) |
|---|---|---|---|---|
| Always R | −194.0 ± 7.8 | 195.0 | 1.00 | −196.3 |
| Always D | −195.9 ± 7.8 | 196.9 | 1.00 | −203.2 |

After the +1 shift above, the expected values are −195.3 and −202.2. Both lie within the 95% interval of the MC estimates.

**What is a trajectory?** It is the sequence s₀, a₀, r₁, s₁, … of one episode. It differs from episode to episode because P(s'|s,a) is random: slips and stays happen at random. A stochastic policy, such as 70% R / 30% D, adds randomness in the choice of action too. The RNG seed fixes which random draws happen.

**Why the average works.** V^π(s₀) = E[G | s₀]. Each episode's return G is an independent, unbiased sample of that expectation. By the law of large numbers the sample mean converges to V^π(s₀), with error about σ/√n. This needs no knowledge of P, only sampled experience, which is why Monte Carlo evaluation is a reinforcement learning method. The returns here have a large variance, because of the long "stuck at the wall" episodes. That is why many episodes are needed.

## Task 4: Value iteration

**Optimal.** π\* minimises the expected number of hops from *every* state at the same time. V\*(s) = max_π V^π(s), so no other policy is better in any state.

**Value iteration.** Repeat the Bellman *optimality* backup V_{k+1}(s) = max_a Σ P(s'|s,a)[R + V_k(s')] on all states except the goal, keeping V(goal) = 0. That boundary condition anchors the solution, just as in Task 2, and the absorbing goal must have value 0 because no more cost is paid there. Stop when max|V_{k+1} − V_k| < tol. Here it took 47 iterations to reach a residual below 1e-8. Each sweep spreads the information "how far is the goal" one more step outward.

**From V\* to π\*.** Take the one-step lookahead π\*(s) = argmax_a Σ P(s'|s,a)[R + V\*(s')]. Acting greedily with respect to V\* is optimal.

**Results.** V\*(0,0) = −10.2 on the heatmap.

The following checks I ran separately; they are not in the notebook:
- In the linear-system convention, V\*(0,0) is −11.2. The fixed policies have about −196 and −203, so π\* is about 18 times better, and it is no worse in any state.
- Simulating π\* (`run_many_simulations(env, pi_opt, n_sims=2000)`) gives a mean return of −10.34 and 11.3 steps on average, matching V\*.

The optimal policy (`pi_opt`, printed as a grid):
```
R R D D D
D R D D D
R R D R D
R R R D D
R R R R .
```
- The arrow map uses only R and D. Unlike the fixed policies, it switches direction *per router*: it never pushes into the wall at the last row or column. Where R and D are both reasonable, it picks whichever has the better link at that router, such as a higher intended probability or a lower stay probability.
- V\* falls off smoothly with Manhattan distance to the goal: about −1.3 per hop, from −10.2 at (0,0) over 8 hops. A hop costs more than 1 because each transmission succeeds only with probability 0.5–0.8.
