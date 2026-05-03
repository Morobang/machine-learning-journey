# Upper Confidence Bound (UCB) — Complete Guide

## Table of Contents
1. [The Multi-Armed Bandit Problem](#the-multi-armed-bandit-problem)
2. [The Exploration vs Exploitation Trade-off](#the-exploration-vs-exploitation-trade-off)
3. [How UCB Works](#how-ucb-works)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Advantages and Disadvantages](#advantages-and-disadvantages)
6. [UCB vs Thompson Sampling](#ucb-vs-thompson-sampling)
7. [Real-World Applications](#real-world-applications)
8. [Implementation Steps](#implementation-steps)
9. [Common Pitfalls](#common-pitfalls)

---

## The Multi-Armed Bandit Problem

Imagine a row of slot machines (called "one-armed bandits" in casinos). Each machine has an unknown probability of paying out. You have a fixed number of pulls. How do you maximise your total reward?

**The dilemma:**
- You could keep pulling the machine that has paid out most so far (**exploit** your current knowledge)
- Or you could try machines you haven't pulled much (**explore** to gather more information)

This is the **multi-armed bandit problem** — the foundational problem in reinforcement learning for sequential decision-making under uncertainty.

**Real advertising example:** You have 10 different ad designs. Each time a user visits your website, you show them one ad and observe whether they click (reward = 1) or not (reward = 0). You don't know in advance which ad has the highest click-through rate. You want to maximise total clicks over N users.

---

## The Exploration vs Exploitation Trade-off

This is the central tension in reinforcement learning:

| Strategy | Description | Risk |
|----------|-------------|------|
| **Pure exploitation** | Always pick the current best option | Miss a better option you haven't tried enough |
| **Pure exploration** | Try all options equally | Waste trials on known bad options |
| **Optimal** | Balance the two intelligently | This is what UCB achieves |

**Why naive approaches fail:**

- **Random selection** — equally explores all options but never exploits what it learns
- **Greedy (always pick current best)** — gets stuck on the first option that was lucky early on, never discovers better options
- **ε-greedy** — explore randomly with probability ε, exploit otherwise — arbitrary ε choice and treats all non-greedy options equally regardless of uncertainty

UCB solves this by being **principled about uncertainty**.

---

## How UCB Works

UCB selects the action (arm/ad) with the highest **upper confidence bound** — the highest plausible reward given current evidence.

### The Intuition
For each option i, we know:
1. The sample mean reward from past trials: x̄ᵢ
2. How many times it has been tried: nᵢ

An option with a high sample mean is good to exploit. But an option that has been tried only once has high uncertainty — its true mean could be much higher. UCB explicitly adds a bonus for uncertainty.

### The Formula
At round n, select the option i that maximises:

```
UCB(i) = x̄ᵢ + √(2 × ln(n) / nᵢ)
```

Where:
- `x̄ᵢ` = average reward from option i so far
- `n` = total number of rounds played so far
- `nᵢ` = number of times option i has been played

**First term (x̄ᵢ):** Reward estimate — the exploitation term
**Second term (√(2 ln n / nᵢ)):** Uncertainty bonus — the exploration term

### Why This Works
- Options tried few times (small nᵢ) have a large uncertainty bonus → they get explored
- As an option is tried more (nᵢ increases), its uncertainty bonus shrinks → it only gets selected if its mean is genuinely high
- As total rounds n increases (ln n grows), even well-tried options get re-explored periodically
- The algorithm automatically reduces exploration as it gains confidence

---

## Mathematical Foundation

### The Hoeffding Bound
The UCB formula comes from **Hoeffding's inequality**, which bounds the probability that a sample mean deviates far from its true mean:

```
P(x̄ᵢ ≥ μᵢ + u) ≤ e^(-2nᵢ u²)
```

Setting the right-hand side equal to 1/n and solving for u gives:

```
u = √(ln(n) / (2nᵢ))
```

This is the confidence bound: with high probability (1 − 1/n), the true mean lies below `x̄ᵢ + √(ln(n) / (2nᵢ))`.

By always selecting the option whose upper confidence bound is highest, we ensure that:
1. We never permanently ignore an option (its UCB grows as n increases even if nᵢ stays fixed)
2. We concentrate on options that have demonstrated high rewards as we get more data

### Regret Bound
UCB has a provably sublinear **regret** — the difference between total reward obtained vs the theoretical best:

```
Regret = O(√(K × n × ln(n)))
```

Where K is the number of arms. This means average regret per round goes to zero — UCB asymptotically converges to always selecting the best arm.

---

## Advantages and Disadvantages

### Advantages

| Advantage | Detail |
|-----------|--------|
| Principled exploration | Uncertainty bonus is theoretically justified, not arbitrary |
| Deterministic | Same data always produces same decisions |
| Provable regret bounds | Mathematically guaranteed to converge to the best arm |
| No hyperparameters | The formula requires no tuning (unlike ε-greedy which needs ε) |
| Simple to implement | Just a formula applied at each round |

### Disadvantages

| Advantage | Detail |
|-----------|--------|
| Deterministic | Cannot inject randomness when determinism is problematic (e.g., adversarial environments where opponents can predict your choices) |
| Assumes stationary rewards | If the true click-through rate changes over time, UCB's historical average becomes stale |
| Must try each arm at least once | Requires an initial round-robin warm-up period |
| Bounded rewards assumed | The Hoeffding bound requires rewards in a fixed range |

---

## UCB vs Thompson Sampling

Both solve the multi-armed bandit problem. The key difference is in how they represent and use uncertainty:

| Property | UCB | Thompson Sampling |
|----------|-----|-------------------|
| Approach | Deterministic — always pick highest UCB | Probabilistic — sample from posterior, pick highest sample |
| Uncertainty representation | Point estimate + confidence interval | Full probability distribution (posterior) |
| Randomness | None | Inherent (stochastic policy) |
| Performance | Strong theoretical bounds | Often better empirically, especially early on |
| Computation | Simple arithmetic | Beta distribution sampling (still cheap) |
| Adversarial environments | Predictable (exploitable) | Randomised (harder to exploit) |

**In practice:** Thompson Sampling often outperforms UCB in A/B testing and ad optimisation because its randomness naturally balances exploration and exploitation without requiring UCB's "burn-in" phase where each arm must be tried once.

---

## Real-World Applications

**Online Advertising:**
Select which ad to display to each visitor to maximise click-through rate. Each ad is an "arm". UCB automatically shifts budget toward higher-performing ads while continuing to test the others.

**Clinical Trials:**
Adaptively allocate more patients to more effective treatments as evidence accumulates. Reduces harm to trial participants compared to fixed equal-allocation designs.

**Recommendation Systems:**
Select which item to recommend to a user. Each item is an arm, reward = click/purchase. UCB balances showing known popular items vs exploring items with few impressions.

**Network Routing:**
Select the routing path with the best throughput. Network conditions change, so UCB continuously re-explores less-used paths.

**E-mail Subject Line Testing:**
Test different subject lines to maximise open rates. UCB shifts more sends toward the winning subject line while it tests alternatives.

---

## Implementation Steps

```python
import numpy as np

def ucb_simulation(n_rounds, n_ads, actual_click_rates):
    """
    Simulate UCB for ad selection.
    actual_click_rates: true CTR for each ad (unknown to the algorithm)
    """
    n_selections = np.zeros(n_ads)    # how many times each ad was shown
    sum_rewards = np.zeros(n_ads)     # total clicks per ad
    total_reward = 0
    ads_selected = []

    for n in range(1, n_rounds + 1):
        # On first pass, play each arm once
        if n <= n_ads:
            ad = n - 1
        else:
            # Compute UCB for each ad
            ucb_values = (sum_rewards / n_selections) + np.sqrt(2 * np.log(n) / n_selections)
            ad = np.argmax(ucb_values)

        # Simulate reward
        reward = np.random.binomial(1, actual_click_rates[ad])

        n_selections[ad] += 1
        sum_rewards[ad] += reward
        total_reward += reward
        ads_selected.append(ad)

    return ads_selected, total_reward

# Example usage
click_rates = [0.1, 0.15, 0.08, 0.20, 0.12]  # ad 3 (index 3) is best
ads, total = ucb_simulation(n_rounds=1000, n_ads=5, actual_click_rates=click_rates)
print(f"Total clicks: {total}")
print(f"Ad selection counts: {np.bincount(ads)}")  # Should show ad 3 selected most
```

---

## Common Pitfalls

**1. Not initialising with one trial per arm**
If nᵢ = 0, the UCB formula divides by zero. Always play each arm at least once before applying UCB.

**2. Applying UCB to non-stationary rewards**
If click-through rates change over time (seasonal effects, trend shifts), UCB's historical average becomes misleading. Use discounted UCB or sliding-window UCB variants for non-stationary problems.

**3. Treating the simulation result as proof of which ad is best**
UCB identifies the best arm probabilistically. Always compute confidence intervals on the final click-through rate estimates before declaring a winner.

**4. Confusing exploration in UCB with random exploration**
UCB explores strategically — it explores options with high uncertainty, not random ones. This is why it has provable efficiency guarantees that ε-greedy does not.

**5. Using UCB for problems with continuous action spaces**
UCB is designed for a finite set of discrete options (K arms). For continuous action spaces, use other RL algorithms (policy gradient, Q-learning).
