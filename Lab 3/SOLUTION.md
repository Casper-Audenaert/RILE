# Lab 3: Reinforcement Learning in FrozenLake (solution notes)

## What was changed in the notebook

All algorithms were already implemented. The only addition is one line at the end of Task 4:

```python
plot_value_heatmap_from_Q(Q_off, title="max_a Q(s,a) — offline Q-learning")
```

The handout asks for a heatmap of max_a Q(s,a). The function `plot_value_heatmap_from_Q` existed in the notebook but was never called.

Setup: `gymnasium==1.0.0` and `pygame` were installed. The newest gymnasium does not import on Python 3.10.

## Task 1: True value of the random policy

**What.** The code solves (I − Q) v = r over the 10 non-terminal states. Q is the transition matrix among the transient states under π(a|s) = ¼. r(s) is the one-step probability of entering G.

**Why it is a probability.** With γ = 1 the return is 1 if the episode ends in G and 0 if it ends in a hole. So V^π(s) = E[G] = Pr(reach G before a hole | s). For example, V(3,2) = 0.44: from the cell next to the goal, a random walker succeeds 44% of the time. V(start) = 0.014.

**Holes and goal are 0.** They are absorbing, and no reward is collected *after* entering them. The +1 is paid on the transition *into* G, so it counts toward the value of the state before G.

**Higher values.** States near G and far from holes have higher values: (3,2) = 0.44, (3,1) = 0.18, (2,2) = 0.14. The values decay toward the start, because every extra step is another chance to slip into a hole.

**Properness.** The random policy is proper. From every state there is a path with positive probability to a terminal state, so the walk is absorbed with probability 1. This makes the spectral radius ρ(Q) < 1 and (I − Q) invertible. If (I − Q) were singular, some set of transient states would trap the walk forever with positive probability, the policy would be improper, and the values would not be uniquely defined.

## Task 2: Monte Carlo vs. TD(0)

**MC (first-visit).**
- Run a full episode.
- Go backwards through it, computing G = γG + r.
- At the *first* visit of each state, add G to that state's running average.
- The result is an unbiased estimate of V. It has high variance (G is 0 or 1), and it can only update after the episode ends.

**TD(0).**
- After *every* step, move V(s) toward the bootstrapped target r + γV(s'): V(s) ← V(s) + α(target − V(s)).
- It learns online and has lower variance, but the target is biased while V(s') is still wrong.
- Information spreads backwards from G one step at a time.

**Results (MSE vs. the true V, log scale, average of 5 trials).**
- **Constant α = 0.2.** Fast at first, then it stalls at about 2–5·10⁻³ and oscillates. A constant step keeps reacting to every new noisy sample, so the estimate wanders around V^π with variance proportional to α. It never reaches zero error.
- **Constant α = 0.05.** A lower noise floor, at about 3–5·10⁻⁴, but a slower start.
- **α₀/(1+N).** Very smooth but slow, at about 5·10⁻⁴ after 10k episodes. The steps shrink so fast that later data barely moves the estimates.
- **α₀/√(1+N).** The best compromise, at about 2–3·10⁻⁴. It still satisfies the Robbins–Monro conditions (Σα = ∞, Σα² < ∞), so it converges, and it is faster than 1/N.
- **MC against TD (inverse, α₀ = 5).** Both reach about 1–1.5·10⁻⁴. MC is slightly smoother later on. In this small, episodic problem MC does as well as TD.

**Which is faster or more stable.** TD with a large constant α falls fastest at first, and decreasing step sizes (or MC) are the most stable at the end.

**Continuing tasks.** TD(0), because it needs no episode end. MC must wait for a return G that is never finished in a continuing task.

## Task 3: SARSA vs. Q-learning (ε = 0.1, α = 0.1, 10k episodes)

**SARSA.** Choose a ~ ε-greedy(Q). Step and observe r, s'. Choose a' ~ ε-greedy(Q). Update Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]. Then set s, a ← s', a'.

**Q-learning.** Choose a ~ ε-greedy(Q). Step. Update Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)].

**On-policy vs. off-policy.** SARSA's target uses the action the behaviour policy *actually* takes next. It therefore learns Q of the ε-greedy policy it follows, which makes it **on-policy**. Q-learning's target uses the greedy action whatever is executed. It learns Q\* of the greedy *target* policy while following ε-greedy, which makes it **off-policy**.

**Results.**
- Q-learning rises much earlier, reaching about 0.7 success by around 1000 episodes. SARSA needs about 2000.
- Both then level off at about 0.65–0.75. That is close to the optimum of 0.82; each greedy evaluation uses only 300 episodes, so the measurements are noisy.
- Q-learning's curve shows deeper dips. The max operator chases noisy Q-values and the greedy policy flips between actions. SARSA averages over the ε-greedy next action and is smoother.

**Effect of ε and α.**
- A larger ε explores more and finds the goal sooner, but SARSA then learns a more cautious policy for the ε-greedy agent, and the data is noisier.
- A larger α learns faster but is noisier and less stable, especially with slippery transitions. A smaller α is stable but slow.

## Task 4: Offline Q-learning from random data

- **Behaviour policy.** The uniform random policy collected 1000 episodes, 7920 transitions. Q-learning then ran 10 epochs over this fixed buffer and never interacted with the environment again.
- **Is it a good behaviour policy?** It is good for *coverage*: every (s,a) gets probability ¼, so every action is tried in every reachable state, and this is what Q-learning needs to learn Q\*. It is poor for *efficiency*: only about 1.4% of episodes reach G, so few transitions near the goal are seen. A good behaviour policy covers all relevant (s,a) pairs, especially those the optimal policy uses, while still reaching rewarding states often. Examples are an ε-greedy version of a decent policy, or a mixture of exploration and expert data.
- **From Q to a policy.** π_Q(s) = argmax_a Q(s,a). Acting greedily means picking, in each state, the action with the highest Q-value. For V\* we take V(s) = max_a Q(s,a).
- **Results.**
  - The greedy policy is **exactly the optimal policy**. I checked this separately with value iteration on `env.unwrapped.P`; that check is not in the notebook. It has V^{π_Q}(start) = 0.8235 = V\*(start), compared with 0.014 for the random policy, and max |V^{π_Q} − V\*| ≈ 4e-11 over all states.
  - The heatmap of max_a Q underestimates V\* by up to 0.22. Q-values are learned slowly (α = 0.1, 10 epochs), but the *ranking* of actions is already right, and the greedy policy only needs the ranking.
  - The optimal policy looks odd at first sight. In the top row it chooses UP or LEFT, pushing into the wall on purpose. On slippery ice the agent then only slips sideways along the wall and can never slip down into a hole. This "hug the wall" behaviour is why success is 82% instead of the random policy's 1.4%.
