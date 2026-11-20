# Lab 3: Presentation prep

## 0. Explained simply

### What's in this folder

| File | What it is |
|---|---|
| `Reinforcement_Learning___Lab_3-4.pdf` | The handout: the tasks and questions. |
| `RL_LAB_3.ipynb` | The notebook: FrozenLake, policy evaluation (exact, MC, TD), SARSA, Q-learning and offline Q-learning. |
| `SOLUTION.md` | What we changed in the code, the results, and why it works. |
| `PREP.md` | This file: what to say and what you might be asked. |

### The concepts in everyday words

- **FrozenLake.** You walk across a frozen lake to reach a goal. There are holes in the ice. The ice is slippery: you go where you want only 1/3 of the time, otherwise you slide to one side. Reaching the goal gives +1; falling in a hole ends the game with 0.
- **Value with γ = 1.** Because the only reward is the +1 at the end, the value of a square is simply the chance of reaching the goal from there. V(start) = 0.014 means a random walker makes it 1.4% of the time.
- **Monte Carlo (MC).** Walk a whole episode, see if you made it, then update every square you visited toward that result. Like grading a student only on the final exam: fair, but you wait until the end, and one exam is noisy.
- **TD(0) (temporal difference).** Update after every single step, using your current guess for the next square: "my estimate here should be about what I think of the square I just landed on." Like updating your arrival time at every traffic light instead of only when you get home. Faster feedback, but it leans on guesses that may still be wrong.
- **Step size α.** How much you trust the newest experience. Big α: learns fast but jumps around. Small or shrinking α: slow, but settles on the right answer.
- **Q(s, a).** The value of *a specific move* in a square, not just of the square: "if I press RIGHT here and play well afterwards, what's my chance of reaching the goal?" With Q you can pick moves directly.
- **ε-greedy behaviour.** Usually take the move with the best Q, sometimes a random one, so you keep discovering things.
- **SARSA vs. Q-learning.**
  - **SARSA** learns about what it *actually* does, clumsy random moves included. It becomes careful.
  - **Q-learning** learns about what it *would* do if it always played its best. It is more optimistic and picks up good routes sooner, but its learning curve is jumpier.
  - Classic picture: next to a cliff, SARSA walks a safe path away from the edge; Q-learning walks right along the edge, because that is optimal if you never misstep.
- **On-policy / off-policy.** On-policy (SARSA): learn about the same behaviour you are using. Off-policy (Q-learning): behave one way (with exploration) but learn about another (the best way).
- **Offline RL.** Learn only from a logbook someone else recorded, here 1000 episodes of a random walker, without ever touching the lake yourself. It works because Q-learning only needs to see every move tried in every square at least now and then.
- **Greedy policy from Q.** In each square, choose the move with the highest Q. Here that succeeds 82% of the time, the best possible on this lake.
- **The strange wall trick.** The best policy sometimes walks *into a wall* on purpose. On slippery ice that is safe: you can only slide along the wall, never toward the hole beside you.

### The whole lab in one paragraph

On a slippery lake, a random walker reaches the goal 1.4% of the time. We computed those chances exactly, then showed that MC (learn after each walk) and TD (learn after each step) both learn the same numbers from experience alone. Then we let the agent learn which moves to make: SARSA and Q-learning both reach about 70% success. Finally, Q-learning on nothing but old random-walk data finds a policy that succeeds 82% of the time, the best possible.

## 1. Opening (30 seconds)

> "FrozenLake 4×4, slippery: every move goes in the intended direction or 90° to either side, each with probability 1/3. Reaching the goal gives +1; holes end the episode with 0. With γ = 1 the value of a state is the probability of reaching the goal from it. First we evaluated the random policy: exactly, then with MC and TD(0). Then we learned control with SARSA and Q-learning, and finally with offline Q-learning from random data."

## 2. Demo flow

| Step | Show | Say |
|---|---|---|
| 1 | Random-walk animation | "Slippery: the agent doesn't go where it intends." |
| 2 | Task 1 heatmap | "V(start) = 0.014: a random walker succeeds about 1.4% of the time. Next to the goal it is 0.44. Holes and the goal are 0." |
| 3 | MC snapshots | "Values stay 0 until successful episodes arrive, then brighten from the goal backwards." |
| 4 | TD(0) snapshots, change `SCHEDULE` | "TD updates after every step using V(s') as the target." |
| 5 | MSE plot (schedules) | "Constant α = 0.2 is fast but noisy; 1/(1+N) is smooth but slow; 1/√(1+N) is the best trade-off." |
| 6 | MC vs. TD plot | "Both reach ~1e-4. MC is fine here because episodes are short." |
| 7 | SARSA vs. Q-learning plot | "Q-learning rises earlier (~1000 episodes vs. ~2000); both level off around 0.7. Q-learning dips more." |
| 8 | Task 4 plots | "Trained only on random data, yet the greedy policy reaches 0.82, the optimal value. max_a Q underestimates the values, but the action ranking is right." |

## 3. Formulas to know by heart

- Value: V^π(s) = E[Σ γ^t R_{t+1}]. With γ = 1 this is Pr(reach G | s).
- Exact evaluation: (I − Q)v = r over transient states.
- First-visit MC: V(s) = average of the returns G from the first visit to s in each episode.
- TD(0): V(s) ← V(s) + α[r + γV(s') − V(s)]. The bracket is the TD error δ.
- SARSA: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)], with a' ~ ε-greedy.
- Q-learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]
- Robbins–Monro conditions: Σα_t = ∞ and Σα_t² < ∞.

## 4. Numbers to remember

- Random policy: V(start) = 0.014, V(cell left of G) = 0.44.
- TD final MSE: const 0.2 ≈ 2–5e-3 (noisy), const 0.05 ≈ 3–5e-4, 1/(1+N) ≈ 5e-4, 1/√(1+N) ≈ 2–3e-4.
- MC vs. TD (inverse, α₀ = 5): both ≈ 1–1.5e-4.
- SARSA and Q-learning level off around 0.65–0.75 success.
- Offline: 1000 random episodes give 7920 transitions. The greedy policy reaches V(start) = 0.82, which equals the optimal value.

## 5. Likely questions and answers

### Task 1

**What does V^π(1,2) mean?**
The probability that, starting from that cell and acting randomly, the agent reaches the goal before falling into a hole. Why: with γ = 1 the return is exactly 1 if the episode ends in G and 0 otherwise. So its expectation equals the probability of success.

**Why are holes and the goal 0?**
They are terminal, so no more reward can be collected *from* them. The +1 is earned on the step *into* G, so it counts toward the cell before G.

**What does the random policy do, and why is it proper?**
It picks each of the 4 actions with probability 1/4. From every cell there is a positive-probability path to a hole or the goal, so the episode ends with probability 1.

**What if (I − Q) were singular?**
Then some set of non-terminal states would keep the agent forever with positive probability: the policy would be improper. The values would not be uniquely defined.
Example: always UP in the top row. The agent slips only left or right along the wall and never leaves the row.

**Which states have high values, and why?**
Those close to G and away from holes, e.g. (3,2) = 0.44. Every extra step is another chance to slip into a hole, so values fall off quickly toward the start.

### Task 2

**MC vs. TD(0): when and how do they update?**
- **MC:** waits until the episode ends, then updates each first-visited state toward the full return G. G is unbiased but has high variance (0 or 1).
- **TD(0):** updates after every step toward r + γV(s'). It uses the current estimate of the next state (bootstrapping), so it has lower variance but is biased while V(s') is wrong.

**Why does constant-α TD oscillate and stay above zero error?**
The step size never shrinks, so each new noisy sample keeps moving the estimate. It fluctuates around the true value with variance proportional to α. A larger α means faster learning but a higher noise floor.

**What do decreasing step sizes do?**
- They average out the noise. If Σα = ∞ and Σα² < ∞, TD converges to the true values.
- 1/(1+N) shrinks too fast: it is stable but slow, because the early wrong estimates take a long time to correct.
- 1/√(1+N) shrinks more slowly: faster progress, and it still converges.

**Which is faster early, and which is more stable late?**
TD with a large constant α falls fastest at first. Decreasing-α TD and MC are the most stable at the end.

**Which method suits a continuing task?**
TD. It needs no episode end. MC needs the complete return G, which never finishes in a continuing task.

**First-visit vs. every-visit MC?**
First-visit uses only the first occurrence of a state in each episode, so its samples are independent and unbiased. Every-visit uses all occurrences: slightly biased, but it uses more data. Both converge.

**Why does the MC grid stay at zero early on?**
Rewards are sparse. Until an episode actually reaches G, every return is 0.

### Task 3

**What are the main steps of SARSA and Q-learning?**
- **SARSA:** choose a ε-greedily → step → choose a' ε-greedily → update toward r + Q(s',a') → move on with (s,a) = (s',a').
- **Q-learning:** choose a ε-greedily → step → update toward r + max Q(s',·) → repeat.

**Which is on-policy and which off-policy, and why?**
- **SARSA is on-policy:** its target uses the action it will *actually* take, so it learns the value of the ε-greedy policy it follows.
- **Q-learning is off-policy:** its target uses the greedy action, whatever the agent does next. It learns Q\* of the greedy policy while behaving ε-greedily.

**Why does Q-learning rise earlier?**
The max in its target spreads good values backwards immediately, without waiting for the ε-greedy agent to actually follow the good path. SARSA's target includes the exploratory actions, so its estimates grow more slowly.

**Why are Q-learning's curves spikier?**
The max picks whichever action is currently overestimated (maximisation bias). Small changes in Q flip the greedy action, so the evaluated success rate jumps. SARSA averages over the next action and is smoother.

**Why does success level off around 0.7 and not 0.82?**
- With constant α = 0.1, Q keeps fluctuating and never fully converges.
- Each evaluation uses only 300 episodes (standard error ≈ 0.03).
- SARSA learns the best policy *for an agent that explores with ε*, which can differ from the best greedy policy.

**Effect of ε?**
- A larger ε explores more and finds the goal sooner, but the data is noisier and the greedy policy changes less often.
- A smaller ε exploits more and learns more slowly.
- To converge to the optimal policy, ε must eventually go to 0 while every action is still tried infinitely often (the "GLIE" condition).

**Effect of α?**
A larger α learns faster but is noisier, especially with slippery transitions. A smaller α is stable but slow.

**Why random tie-breaking?**
At the start all Q are 0. `argmax` would always pick action 0 (LEFT), which biases exploration. Random ties spread the early choices across actions.

**Cliff-walking intuition (a classic follow-up).**
SARSA learns a *safer* path because it accounts for its own random moves. Q-learning learns the optimal path but walks along the edge and falls more often *during training*.

### Task 4

**What is the behaviour policy here?**
The uniform random policy that collected the data: 1000 episodes, 7920 transitions. After that, learning never touches the environment again.

**Is it a good behaviour policy?**
- **Coverage:** good. Every action is tried in every reachable state, and that is what Q-learning needs.
- **Efficiency:** poor. Only about 1.4% of episodes reach G, so there is little data near the goal.

A good behaviour policy covers all relevant state–action pairs (especially those the optimal policy uses) and reaches rewarding states often. Examples: ε-greedy around a decent policy, or a mix of expert and exploratory data.

**Why does off-policy learning work from random data?**
The Q-learning target max_a' Q(s',a') does not depend on which action the behaviour policy took next. As long as every (s,a) appears in the data, the updates move Q toward Q\*.

**What can go wrong with offline RL in general?**
Missing coverage. If an action is never tried in a state, its Q-value is just a guess, and the max can pick an overestimated, unseen action ("distribution shift / extrapolation error"). Here the table is small and fully covered, so it works.

**How do we get a policy from Q?**
π(s) = argmax_a Q(s,a): in each state, take the action with the highest Q-value. Acting greedily means following that policy with no randomness.

**Is the greedy policy optimal?**
Yes. Its true value at the start is 0.8235, which equals V\*(start). I checked this with value iteration on the true model (not in the notebook).

**Why does the max_a Q heatmap show lower values than the true-value heatmap?**
Q is still underestimated after 10 epochs with α = 0.1. The *ranking* of the actions is already correct, and the greedy policy only needs the ranking.

**The optimal policy looks strange. Why?**
In the top row it presses UP, *into the wall*. On slippery ice the agent then only moves left or right along the wall or stays put, so it can never slide down into the holes in row 1. It drifts to the top-left corner. There LEFT can only slip *down*, into the safe cell (1,0), never right toward the hole at (1,1). The same idea repeats lower down: the agent picks actions whose possible slips avoid the holes. That is why it succeeds 82% of the time instead of 1.4%.

## 6. Traps to avoid

- γ = 1 is fine *only* because episodes always terminate and returns are bounded (0 or 1).
- Action order in FrozenLake is 0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP.
- "Q-learning learns the optimal Q, not a policy": the policy comes from the argmax afterwards.
