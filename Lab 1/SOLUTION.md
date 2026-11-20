# Lab 1: Bandits and Contextual Bandits (solution notes)

## What was changed in the notebook

| Cell | Change |
|---|---|
| Task 1 (a), ε-greedy | Replaced the placeholder with the ε-greedy rule |
| Task 1 (b), UCB | Replaced the placeholder with the UCB rule |
| Part 2 | Unchanged; it only has to be run |

The other experiments in the handout come from changing the parameters at the bottom of the existing cells and rerunning them:

| Experiment | What to change |
|---|---|
| ε = 0.01, 0.1, 0.2 | `epsilon` in the ε-greedy cell |
| c = 1, 2, 5 | `c` in the UCB cell |
| Random policy | `epsilon = 1.0` in the ε-greedy cell |
| Handout horizon | `T = 1000` |

### ε-greedy

```python
if rng.random() < epsilon:
    arm = int(rng.integers(0, k))
else:
    arm = int(np.argmax(Q))
```

With probability ε the agent explores by pulling a uniformly random arm. Otherwise it exploits by pulling the arm with the highest estimated mean `Q`. The update that follows is already in the starter code:
`Q[a] += (r - Q[a]) / N[a]`. That update is the incremental form of the sample mean, so `Q[a]` is always the average reward observed from arm `a`.

### UCB

```python
ucb = Q + c * np.sqrt(np.log(t) / np.maximum(N, 1))
ucb[N == 0] = np.inf
arm = int(np.argmax(ucb))
```

Each arm gets a score: its estimated mean plus an uncertainty bonus `c·sqrt(ln t / N(a))`. Arms that have never been pulled get +∞, so each arm is tried once first. `np.maximum(N, 1)` only prevents a division by zero; those entries are overwritten with ∞ on the next line anyway.

### Random policy

With `epsilon = 1.0` the agent always explores, so the ε-greedy function becomes exactly the uniform random policy, and no separate function is needed.

## Results (probs = [0.2, 0.1, 0.5, 0.9], T = 1000, seed 123)

This is one example run. The environment draws clicks with numpy's global random generator, which the `seed` parameter does not control. Your numbers will therefore differ a little between runs, but the pattern stays the same.

| Policy | Avg reward | Cum. regret | Pulls per arm [0,1,2,3] |
|---|---|---|---|
| Random | 0.418 | 476.1 | [231 252 282 235] |
| ε-greedy ε=0.01 | 0.350 | 546.5 | [755 21 3 221] |
| ε-greedy ε=0.1 | 0.865 | 47.1 | [29 20 27 924] |
| ε-greedy ε=0.2 | 0.810 | 104.6 | [58 54 52 836] |
| UCB c=1 | 0.863 | **34.0** | [8 8 55 929] |
| UCB c=2 | 0.812 | 77.8 | [38 28 72 862] |
| UCB c=5 | 0.648 | 242.6 | [118 108 184 590] |

Part 2 (T = 20 000):

| Policy | Final avg reward | Cum. regret |
|---|---|---|
| Random | 0.790 | 3440.2 |
| Contextual ε-greedy | 0.940 | 501.6 |
| LinUCB | **0.961** | **56.1** |

## Why it works and how to interpret it

**Random.** Its regret grows linearly: each step loses on average µ* − mean(µ) = 0.9 − 0.425 ≈ 0.475, so 1000 steps give about 475. It never learns anything.

**ε-greedy.**
- **ε = 0.1** finds arm 3 quickly and uses it about 92% of the time. Regret still grows linearly, but slowly: about ε·(0.9 − 0.425) ≈ 0.05 per step, because it keeps exploring at a fixed rate forever.
- **ε = 0.2** explores twice as much, so its regret is about twice as high.
- **ε = 0.01** shows how too little exploration fails. All `Q` start at 0, so `argmax` breaks ties toward arm 0. Arm 0 pays 0.2 now and then, which keeps its `Q` above 0, so the agent stays on it. It only rarely explores and finds arm 3. The result is worse than random in this run. This is why ε must be large enough, or Q should be initialised optimistically, or ε should decay over time.

**UCB.**
- The bonus `sqrt(ln t / N)` is large for arms that have rarely been pulled and shrinks as `N` grows, while `ln t` grows only slowly. A bad arm is therefore re-tried less and less often: only O(log T) times. This is called *optimism in the face of uncertainty*, and it gives logarithmic regret instead of linear regret. In the plot, the regret curve flattens out.
- **c = 1** gives the lowest regret. With a larger c (2, 5) the bonus dominates the estimated mean for longer, so UCB keeps checking the bad arms. c = 5 behaves almost randomly during the first 1000 steps.
- Compared with ε-greedy, UCB explores in a *directed* way: it explores arms because they are uncertain, not by chance. Clearly bad arms (0 and 1) are pulled only 8 times each. ε-greedy wastes about the same number of pulls on every arm, good or bad.

**Regret** can only be computed here because the simulator knows µ. In a real system you only see the rewards. Regret is still the standard way to compare how efficient bandit algorithms are.

## Part 2: Contextual bandits

**How context affects the choice.** The click probability is σ(θ_aᵀx), so the best article depends on the user. In the printed examples, techies get Tech, sports fans get Sports, and older morning readers interested in politics get Politics. A context-free bandit can only learn one "best on average" article.

**Contextual ε-greedy.** It keeps a linear model θ̂_a for each arm, which is updated with an SGD step `θ̂_a += (r − θ̂_aᵀx)·x / N_a`. It picks argmax θ̂_aᵀx, or a random arm with probability ε. It uses the context only through the point estimate, and it explores blindly at rate ε forever. That is why its regret keeps growing roughly linearly at about 0.025 per step.

**LinUCB.** For each arm it solves a ridge regression: θ̂_a = A_a⁻¹ b_a, with A_a = λI + Σ xxᵀ. It adds the bonus α·sqrt(xᵀA_a⁻¹x), which is the width of the confidence interval of the prediction *for this particular user x*. The bonus is large when a user looks like nobody this arm has seen before, and small for well-covered user types. Exploration therefore targets the parts of the feature space where we know little. This is why LinUCB learns faster and its regret curve flattens (56 against 502).

**Before vs. after learning (Cell 3).** In the first steps, ε-greedy has all scores at 0 and picks almost at random. LinUCB's choices are driven by the bonus terms, which are all large and nearly equal. After 10 000 samples both mostly pick the true best arm. LinUCB's bonuses have shrunk to about 0.03–0.2, so its predictions now drive the choice. In "after" step 4, ε-greedy still picks Sports for a politics reader. LinUCB picks Politics.

**Role of the hyperparameters.**
- **ε**: the fraction of random exploration.
- **α**: how wide the confidence bonus is. A larger α explores more.
- **λ**: the ridge prior/regularisation. It keeps A invertible, shrinks θ̂ toward 0, and makes the initial bonus larger.

**Linearity assumption.** The true reward is σ(θᵀx), which is logistic, not linear. A linear model fitted to 0/1 rewards is therefore misspecified. It still mostly works because σ is monotone: the arm with the largest θᵀx also has the largest σ(θᵀx), so a linear fit tends to keep the *ranking* of arms, and the ranking is all the policy needs. It goes wrong near the edges, where σ saturates. Better options:
- a GLM/logistic bandit (logistic-UCB, GLM-UCB) or Thompson sampling with logistic regression, which match the true model;
- richer features or interactions, or neural or kernel bandits, if the true model is unknown.
