# Thompson Sampling — Complete Guide

## Table of Contents
1. [What is Thompson Sampling?](#what-is-thompson-sampling)
2. [Bayesian Inference Primer](#bayesian-inference-primer)
3. [How Thompson Sampling Works](#how-thompson-sampling-works)
4. [Mathematical Foundation: Beta Distribution](#mathematical-foundation-beta-distribution)
5. [Advantages and Disadvantages](#advantages-and-disadvantages)
6. [Thompson Sampling vs UCB](#thompson-sampling-vs-ucb)
7. [Real-World Applications](#real-world-applications)
8. [Implementation Steps](#implementation-steps)
9. [Common Pitfalls](#common-pitfalls)

---

## What is Thompson Sampling?

**Thompson Sampling** is a Bayesian approach to the multi-armed bandit problem. Instead of computing a confidence bound (like UCB), it maintains a **probability distribution** over what the true reward rate might be for each option, and uses random sampling from these distributions to make decisions.

The core idea: **act as if your current probability distribution is the true distribution, then update it as you gather evidence.**

Thompson Sampling was first proposed by W.R. Thompson in 1933 — long before modern reinforcement learning — and has seen a resurgence since 2010 when it was proven to achieve optimal regret bounds. It is now the algorithm of choice for many real-world bandit problems.

---

## Bayesian Inference Primer

Thompson Sampling is grounded in **Bayes' theorem**:

```
P(hypothesis | data) ∝ P(data | hypothesis) × P(hypothesis)
```

- **Prior P(hypothesis):** What we believe before seeing data
- **Likelihood P(data | hypothesis):** How probable is the data given the hypothesis?
- **Posterior P(hypothesis | data):** Updated belief after seeing data

For the ad click problem:
- **Hypothesis:** The true click-through rate θ for an ad
- **Prior:** Before any data, we assume θ is equally likely to be any value in [0,1] — uniform prior
- **Likelihood:** Each click is a Bernoulli(θ) trial
- **Posterior:** After observing some clicks, we update our belief about θ

The **Beta distribution** is the natural choice for a prior on a probability — it is defined on [0,1] and has the convenient property that it is the **conjugate prior** for the Bernoulli likelihood, meaning the posterior is also a Beta distribution.

---

## How Thompson Sampling Works

For each option i (each ad), maintain a Beta distribution parameterised by (αᵢ, βᵢ):
- **αᵢ** = 1 + number of successes (clicks) from ad i
- **βᵢ** = 1 + number of failures (no-clicks) from ad i

**At each round:**

1. **Sample** one value θᵢ from the Beta(αᵢ, βᵢ) distribution for each ad i
2. **Select** the ad i with the highest sampled θᵢ
3. **Observe** the reward (click or no click)
4. **Update** the distribution:
   - If click (reward = 1): increment αᵢ by 1
   - If no click (reward = 0): increment βᵢ by 1

**Why this works:** An ad with many successes (high α) will have its distribution concentrated near a high value — it will usually sample high. An ad that has barely been tried (low α + β) will have a flat, wide distribution — it occasionally samples very high values, causing it to be explored. Over time, the distributions narrow around the true click-through rates, and the algorithm concentrates on the best ad.

---

## Mathematical Foundation: Beta Distribution

The **Beta(α, β)** distribution has:

```
Mean = α / (α + β)
Variance = (α × β) / ((α + β)² × (α + β + 1))
```

**Initial state — Beta(1, 1) = Uniform(0, 1):**
Before any data, we know nothing about the click rate. The uniform distribution assigns equal probability to every possible rate.

**After gathering data:**
- Beta(10, 2): 10 successes, 2 failures → distribution concentrated near 0.83 (high CTR)
- Beta(2, 10): 2 successes, 10 failures → distribution concentrated near 0.17 (low CTR)
- Beta(1, 1): untried → flat distribution, will sometimes sample very high or low

**As data accumulates (α + β grows):**
The distribution narrows — we become more certain about the true rate. A narrow distribution rarely samples far from its mean, so the algorithm exploits the best option more and more.

**The exploration mechanism:**
A poorly-tried ad has low α + β → wide distribution → occasionally samples above the mean of the current best ad → gets explored. This is automatic — no explicit exploration term needed.

---

## Advantages and Disadvantages

### Advantages

| Advantage | Detail |
|-----------|--------|
| Empirically excellent | Often outperforms UCB, especially in the early rounds |
| Naturally Bayesian | Prior knowledge can be incorporated (e.g., "we expect a 5% CTR based on past campaigns") |
| Handles uncertainty elegantly | The full distribution captures what we know and don't know |
| Randomised policy | Not predictable by adversaries; good for adversarial settings |
| Extends naturally | Beta-Binomial is just the simplest case; extends to Gaussian rewards, contextual bandits, etc. |
| Optimal regret bounds | Provably achieves the Lai-Robbins lower bound for regret |

### Disadvantages

| Disadvantage | Detail |
|-------------|--------|
| Stochastic | Different runs produce different results (though this is also an advantage in adversarial settings) |
| Conjugate prior required for efficiency | The Beta-Binomial simplicity breaks down for complex reward distributions; need approximations |
| Non-stationary challenges | Like UCB, struggles when true rates change over time |
| Less intuitive | The sampling procedure is less immediately obvious than UCB's formula |

---

## Thompson Sampling vs UCB

| Property | Thompson Sampling | UCB |
|----------|-------------------|-----|
| Approach | Sample from posterior distribution | Upper confidence bound formula |
| Exploration | Implicit — wide posteriors sample high occasionally | Explicit — uncertainty bonus added to estimate |
| Randomness | Stochastic — different runs may differ | Deterministic — same data → same decision |
| Empirical performance | Usually better, especially early on | Strong theoretical guarantees |
| Prior knowledge | Natural (Bayesian framework) | Harder to incorporate |
| Complexity | Slightly more complex (distributions) | Simpler (arithmetic formula) |
| Adversarial robustness | Good (randomised policy) | Poor (deterministic) |

**The consensus in practice:** Thompson Sampling is preferred for most real-world applications because:
1. It performs better empirically (less regret in practice)
2. The Bayesian framework naturally incorporates business knowledge via priors
3. The randomised policy is harder to game

UCB has stronger worst-case guarantees and is easier to explain to non-technical stakeholders.

---

## Real-World Applications

**Ad Campaign Optimisation:**
Major technology companies (Google, LinkedIn, Twitter) use Thompson Sampling for online ad selection. It maximises revenue by quickly identifying high-performing ads while continuing to test alternatives.

**A/B/n Testing:**
Traditional A/B testing has a fixed sample size and ignores accumulating evidence. Thompson Sampling performs **adaptive allocation** — sending more traffic to the currently better-performing variant while still testing the other. This reduces harm to users from the inferior variant.

**Clinical Trials (Adaptive Design):**
Adaptive Bayesian clinical trials use Thompson Sampling to allocate more patients to more effective treatments as the trial progresses. This is more ethical than fixed equal allocation and can achieve the same statistical power with fewer total patients.

**Personalised Recommendations:**
With contextual extensions (contextual bandits), Thompson Sampling personalises which recommendation to show each user based on their features — balancing exploration of new items with exploitation of known preferences.

**Game Playing (Heuristic Search):**
Monte Carlo Tree Search uses a variant of Thompson Sampling to decide which game tree branches to explore — a key component of AlphaGo and chess engines.

---

## Implementation Steps

```python
import numpy as np

def thompson_sampling(n_rounds, n_ads, actual_click_rates):
    """
    Simulate Thompson Sampling for ad selection.
    actual_click_rates: true CTR for each ad (unknown to the algorithm)
    """
    # Initialise Beta(1,1) prior for each ad (uniform — no prior knowledge)
    alpha = np.ones(n_ads)   # successes + 1
    beta_params = np.ones(n_ads)   # failures + 1

    total_reward = 0
    ads_selected = []

    for _ in range(n_rounds):
        # 1. Sample theta from each ad's Beta distribution
        theta_samples = np.random.beta(alpha, beta_params)

        # 2. Select the ad with the highest sample
        ad = np.argmax(theta_samples)

        # 3. Observe reward
        reward = np.random.binomial(1, actual_click_rates[ad])

        # 4. Update the distribution
        if reward == 1:
            alpha[ad] += 1
        else:
            beta_params[ad] += 1

        total_reward += reward
        ads_selected.append(ad)

    return ads_selected, total_reward, alpha, beta_params

# Example
click_rates = [0.1, 0.15, 0.08, 0.20, 0.12]  # ad 3 is best
ads, total, alpha_final, beta_final = thompson_sampling(1000, 5, click_rates)

print(f"Total clicks: {total}")
print(f"Ad selection counts: {np.bincount(ads)}")
print(f"Estimated CTRs: {alpha_final / (alpha_final + beta_final)}")

# Incorporate domain knowledge via informative priors:
# If you know from past campaigns that CTR is ~10%, start with Beta(10, 90)
# instead of Beta(1, 1). This guides exploration toward realistic values.
alpha_informed = np.full(5, 10.0)   # prior: expect ~10% CTR
beta_informed = np.full(5, 90.0)    # prior: expect ~10% CTR
```

---

## Common Pitfalls

**1. Ignoring the prior entirely**
Starting with Beta(1, 1) means you explore all options equally before any data. If you have prior knowledge (previous campaign data, industry benchmarks), encode it in the initial α and β parameters. This reduces the exploration phase substantially.

**2. Not distinguishing Thompson Sampling from standard A/B testing**
Traditional A/B testing wastes budget on inferior variants throughout the test period. Thompson Sampling reduces this waste by reallocating dynamically. Do not mix the two approaches — either run a fixed A/B test or run Thompson Sampling; do not switch mid-experiment.

**3. Assuming one run is sufficient**
Thompson Sampling is stochastic. For analysis purposes (which ad is best?), run multiple simulations and take the average performance. For deployment (actual ad serving), the stochasticity is a feature — it provides natural exploration.

**4. Using Beta-Binomial for non-binary rewards**
If rewards are continuous (revenue per click, not just click/no-click), use a different conjugate pair. For Gaussian rewards, use a Normal-Normal model. The Beta distribution is only appropriate for binary (Bernoulli) rewards.

**5. Non-stationary environments**
If click-through rates change (different time of day, seasonal trends), the accumulated α and β from weeks ago may no longer be relevant. Use **discounted updates** — gradually reduce the weight of old observations — or reset periodically.
