# Lab 1: Presentation prep

## 0. Explained simply

### What's in this folder

| File | What it is |
|---|---|
| `Reinforcement_Learning___Lab_1.pdf` | The handout: the tasks and questions. |
| `RL_Lab_1.ipynb` | The notebook. We filled in ε-greedy and UCB and ran every cell. |
| `SOLUTION.md` | What we changed in the code, the result tables, and why it works. |
| `PREP.md` | This file: what to say and what you might be asked. |

### The concepts in everyday words

- **Bandit, arm.** Picture a row of slot machines. Each one pays out with its own hidden chance. Here each machine is an article, and a payout is a click. "Pulling an arm" = showing that article.
- **Exploration vs. exploitation.** Do you go back to your favourite restaurant (exploit) or try a new one that might be better (explore)? Only exploiting can leave you stuck with a mediocre place forever; only exploring means you never enjoy the good one.
- **Q(a).** Our current guess of how good article a is: simply its average click rate so far.
- **Regret.** How many clicks we lost compared with someone who knew the best article from day one. Lower is better. A curve that goes flat means "we stopped making mistakes"; a straight rising line means "we keep making them".
- **ε-greedy.** "Go to your favourite restaurant, but 10% of the time pick one at random." Easy, and it works, but it keeps visiting bad places at random forever, so regret keeps growing.
- **UCB (Upper Confidence Bound).** "Give every option the benefit of the doubt." Score = how good it looked so far + a bonus for how little we know about it. Tried twice → big bonus. Tried 500 times → almost no bonus. So we only re-check options that could still turn out to be the best.
- **Contextual bandit.** Now we know something about the reader (age, interests, time of day). The best article depends on who is reading, like a waiter who recommends a dish based on who sits down.
- **Linear model θ_aᵀx.** For each article, a list of weights: how much each user feature pushes the click chance up or down. The prediction is a weighted sum of the user's features.
- **σ (sigmoid).** Squashes any number into a probability between 0 and 1. The simulator uses it to turn θᵀx into a click chance.
- **Contextual ε-greedy.** Learns those weights, shows the article with the best prediction, and 10% of the time shows a random one.
- **LinUCB.** Same predictions, plus a bonus that says "I haven't seen many readers like *this one* for this article". If an article was never shown to older politics readers, it gets a big bonus for such a reader, and a small one for readers it already knows well.
- **A_a, λ, α in LinUCB.**
  - A_a is a running record of which kinds of readers article a has been shown to.
  - λ is a small "start from zero and be cautious" setting that keeps the maths working before any data arrives.
  - α is how adventurous the agent is: bigger α means more exploring.

### The whole lab in one paragraph

We must keep choosing articles without knowing which ones people like. Picking at random wastes about half the clicks. ε-greedy learns the best article but keeps wasting a fixed share of turns on random picks. UCB explores only where it is still unsure, so it stops wasting turns and its regret flattens. When we also know who the reader is, LinUCB does the same trick per type of reader and beats contextual ε-greedy by a wide margin (regret 56 vs. 502).

### The notebook, cell by cell

| # | Cell | What it does, in one line |
|---|---|---|
| 1 | Intro (text) | The story: we are NewsNet, recommending one article per user; click = 1, no click = 0. |
| 2 | Part 1 intro (text) | Explains the environment: every article has a hidden click chance. |
| 3 | `BernoulliBandit` | The simulator. `pull(arm)` flips a biased coin for that article and returns 1 or 0. |
| 4 | Task 1 (a) text | Asks us to fill in ε-greedy. |
| 5 | `run_epsilon_greedy` | **Our code.** Each step: random article with chance ε, else the best Q. Then updates Q (running average) and plots average reward, regret and how often each article was picked. Change `epsilon` at the bottom. |
| 6 | Task 1 (b) text | Asks us to fill in UCB. |
| 7 | `run_ucb` | **Our code.** Each step: pick the article with the highest "average + uncertainty bonus"; untried ones first. Same three plots. Change `c` at the bottom. |
| 8–9 | Part 2 intro (text) | Now we also see user features (interests, age, time of day). |
| 10 | Cell 1: environment and policies | `ContextualNewsEnv` (click chance = σ(θ_aᵀx), so it depends on the user) and the three policies: Random, contextual ε-greedy, LinUCB. Only defines things; nothing is plotted. |
| 11 | Cell 2: long-run simulation | Runs all three policies on the same 20 000 users and plots average reward and cumulative regret. Main result: LinUCB 56, ε-greedy 502, random 3440. |
| 12 | Cell 3: before vs. after | Prints 6 decisions *before* learning, trains for 10 000 rounds, then prints 6 decisions *after*, with each policy's scores. Shows the bonus shrinking and the choices becoming correct. |

## 1. Opening (30 seconds)

> "We recommend one of k articles to each user. A click gives reward 1, no click gives 0. We don't know the click rates, so we have to explore to learn them and exploit to collect clicks. In Part 1 every user is the same: a multi-armed bandit. We compared ε-greedy and UCB against random. In Part 2 we see the user's features, and we compared contextual ε-greedy with LinUCB."

## 2. Demo flow

| Step | Show | Say |
|---|---|---|
| 1 | `BernoulliBandit.pull` | "Each arm is a coin with an unknown probability. The environment flips it." |
| 2 | ε-greedy code (the 4 lines I wrote) | "With probability ε a random arm, otherwise argmax Q. Q is the sample mean of each arm, updated incrementally." |
| 3 | ε-greedy plots | "The average reward approaches 0.9 and almost all pulls go to arm 3. Regret keeps growing linearly because we keep exploring at a fixed rate ε." |
| 4 | UCB code (3 lines) | "Score = mean + bonus. The bonus is large for arms we have tried rarely. Untried arms get infinity, so each is tried once." |
| 5 | UCB plots | "Regret flattens out (logarithmic). Bad arms are pulled only a handful of times." |
| 6 | Change ε / c / ε=1.0 live if asked | Use the parameter block at the bottom of each cell. |
| 7 | Part 2, Cell 2 plot | "LinUCB has the lowest regret, ε-greedy is in the middle, random has linear regret." |
| 8 | Part 2, Cell 3 printout | Compare one "before" step with one "after" step (see §4). |

## 3. Formulas to know by heart

- Incremental mean: Q_n = Q_{n−1} + (1/n)(r_n − Q_{n−1})
- Regret: Regret_T = Σ_t (µ\* − µ_{a_t})
- UCB: a_t = argmax_a [ Q(a) + c·√(ln t / N(a)) ]
- LinUCB:
  - A_a = λI + Σ x xᵀ
  - b_a = Σ r x
  - θ̂_a = A_a⁻¹ b_a
  - score = θ̂_aᵀx + α√(xᵀA_a⁻¹x)
- Click model: p = σ(θ_aᵀx), with σ(z) = 1/(1+e^(−z))

## 4. Numbers to remember

These are from an example run with T = 1000. Your own run will differ slightly.
- Random: regret ≈ 475. The expected loss per step is 0.9 − mean(0.2, 0.1, 0.5, 0.9) = 0.475.
- ε = 0.1: regret ≈ 47. ε = 0.2: about twice that. ε = 0.01: it can get stuck on arm 0 and do *worse* than random.
- UCB c = 1: the best, regret ≈ 34. c = 5 over-explores (≈ 240).
- Part 2 (T = 20 000): LinUCB regret ≈ 56, contextual ε-greedy ≈ 500, random ≈ 3440.

## 5. Likely questions and answers

**Exploration vs. exploitation: what's the dilemma?**
Exploiting means picking what looks best now. Exploring means gathering information that may reveal something better. Pure exploitation can lock onto a bad arm forever. Pure exploration never uses what it has learned.

**Why is Q(a) updated with 1/N(a)?**
That makes Q(a) exactly the sample mean of the rewards from arm a. By the law of large numbers it converges to µ_a.

**Why does ε-greedy have linear regret?**
A fraction ε of the steps is random no matter how much we already know. Each such step loses on average µ\* − mean(µ). So regret ≈ ε·(µ\* − µ̄)·T. The fix is to let ε decay, e.g. ε_t ∝ 1/t. That gives logarithmic regret.

**Why did ε = 0.01 fail?**
All Q start at 0, and `np.argmax` breaks ties by picking the first index, which is arm 0. Once arm 0 pays out once, Q(0) > 0, so greedy keeps choosing it. With ε = 0.01 the agent explores so rarely that it takes a long time to find arm 3.
Fixes: random tie-breaking, optimistic initial values (Q = 1), or a larger or decaying ε.

**Where does the UCB bonus come from?**
From Hoeffding's inequality. With high probability the true mean lies within √(ln t / N) of the sample mean. UCB acts as if each arm is as good as its *upper* confidence bound ("optimism in the face of uncertainty").
- If an arm is truly good, pulling it earns reward.
- If it is bad, pulling it shrinks its bonus, so we stop.

**Why ln t in the bonus?**
It grows slowly but without limit, so every arm keeps being re-checked occasionally. A bad arm is pulled only O(log T) times, which gives logarithmic regret.

**What does c do? What about c = 0?**
c scales the exploration. c = 0 is pure greedy, with no exploration. A large c (5) keeps testing bad arms for a long time, almost random in the first 1000 steps. A small c risks under-exploring.

**ε-greedy vs. UCB: what's the key difference?**
ε-greedy explores *blindly*: it is equally likely to try a clearly bad arm as an uncertain one. UCB explores *in a directed way*: it only explores arms whose upper bound could still beat the leader.

**Average reward vs. regret: how are they related?**
Expected average reward = µ\* − Regret_T / T. Regret is more informative because it shows whether the agent *stops* making mistakes (a flat curve) or keeps making them (a linear curve).

**Can you measure regret in practice?**
No. It needs the true µ, which only the simulator knows. In practice we look at reward, CTR, A/B tests, or off-policy evaluation.

**What would you use if click rates change over time?**
A constant step size: Q ← Q + α(r − Q). This weights recent rewards more. Alternatives are sliding-window or discounted UCB.

**Another algorithm?**
Thompson sampling. Keep a Beta(1 + clicks, 1 + misses) posterior for each arm, draw one sample per arm, and pick the arm with the largest sample. It is often the best in practice.

### Part 2 questions

**Why use context at all? Why not one bandit per user type?**
The best article depends on the user. The features are continuous and there are many combinations, so one bandit per user would never get enough data. A model θ_aᵀx *generalises*: what we learn from one sports fan helps with all similar users.

**How does contextual ε-greedy use the context?**
It keeps a linear estimate θ̂_a for each arm and predicts θ̂_aᵀx for the current user. It picks the best prediction, or a random arm with probability ε. The update is one SGD step on the squared error: θ̂_a += (r − θ̂_aᵀx)·x / N_a. It has no notion of uncertainty.

**How does LinUCB use it differently?**
It uses x twice:
- for the **prediction** θ̂_aᵀx, from ridge regression;
- for the **uncertainty** √(xᵀA_a⁻¹x), which is how uncertain the model is *for this specific user*.

Arms that have seen many similar users have a small bonus. Arms that rarely saw such users have a large bonus. Exploration therefore goes where knowledge is missing.

**What is A_a intuitively?**
It is the regularised Gram matrix of all contexts on which arm a was shown, so it measures how much data we have in each direction of feature space. A_a⁻¹ is (up to noise) the covariance of θ̂_a. As data comes in, A grows, A⁻¹ shrinks, and the bonus shrinks.

**Role of λ?**
- It makes A invertible at the start, when there is no data.
- It shrinks θ̂ toward 0 (ridge regularisation, equivalent to a Gaussian prior on θ).
- A larger λ gives more conservative estimates.

**Role of α?**
It is the width of the confidence bonus. A larger α explores more; α = 0 is greedy ridge regression.

**Why does LinUCB learn faster than ε-greedy?**
Its exploration is targeted, and its estimate is the exact least-squares solution rather than noisy SGD steps. ε-greedy wastes about 10% of rounds on random arms forever, which shows up as linearly growing regret.

**Explain one before/after example from Cell 3.**
- **Before:** ε-greedy's scores are all 0 or tied, so it picks almost at random. LinUCB's bonuses (~1.3–2.1) dominate its predictions, so it is exploring.
- **After 10 000 samples:** LinUCB's bonuses are ~0.03–0.2 and the predictions decide the choice. Both methods mostly pick the true best arm.
- In "after" step 4, a politics reader, ε-greedy picked Sports and LinUCB picked Politics.

**Is the linearity assumption correct? Can we do better?**
- No. The true model is logistic, σ(θᵀx), and fitting a linear model to 0/1 rewards is misspecified.
- It mostly works because σ is monotone: the arm with the largest θᵀx also has the largest σ(θᵀx), and choosing an arm only needs the *ranking* of the arms.
- It is worst where σ saturates (near 0 or 1).
- Better options:
  - a logistic or GLM bandit (GLM-UCB), or Thompson sampling with logistic regression;
  - richer features, or neural or kernel bandits, if the true form is unknown.

**Computational cost of LinUCB?**
Each step solves a d×d system per arm, O(k·d³). This can be done in O(k·d²) by updating A⁻¹ with the Sherman–Morrison formula. For d = 6 it is trivial.

## 6. Traps to avoid

- Don't say UCB "has zero regret". Its regret grows logarithmically, which is sub-linear, not zero.
- The seed does not make Part 1 fully reproducible, because the environment uses `np.random`, not `rng`. If the numbers differ from your notes, that's why.
- "Regret" here is *expected* (pseudo-)regret: µ\* − µ_{a_t}, not the difference in the rewards actually received.
