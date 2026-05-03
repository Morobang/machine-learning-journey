# Reinforcement Learning

Reinforcement learning (RL) is the branch of machine learning where an **agent** learns to make decisions by interacting with an **environment** and receiving feedback in the form of rewards or penalties. There are no labelled examples — the agent must discover what actions lead to good outcomes through trial and error.

The central metaphor: a child learning to walk. Nobody hands the child a dataset of correct foot movements. Instead, the child tries things, falls, adjusts, and gradually learns the actions that lead to staying upright. The reward is not falling.

---

## The Multi-Armed Bandit Problem

This section focuses on a specific, foundational RL problem: the **multi-armed bandit**.

Imagine a row of slot machines. Each machine has an unknown probability of paying out. You have N total pulls to distribute across the machines. Your goal is to maximise total payout.

This is not a trivial problem. If you spend all your pulls on the machine that paid out first, you might miss the best machine. If you spend all your pulls exploring, you waste time on machines you already know are bad.

This exact structure appears in real systems:
- Showing one of many ad creatives to website visitors — which ad gets the most clicks?
- Allocating patients in a clinical trial to one of several treatments — which treatment helps most?
- Recommending one of many items to a user — which recommendation gets the most engagement?
- Testing different subject lines in an email campaign — which gets the most opens?

The two algorithms in this section both solve the multi-armed bandit problem. They differ in *how* they balance exploration and exploitation.

---

## The Exploration vs Exploitation Trade-off

This is the central tension of reinforcement learning:

**Exploitation** — choose the option with the highest known reward. Safe, but you may miss a better option you have not tried enough.

**Exploration** — try less-tested options to gather more information. Necessary to find the best option, but wastes pulls on options that may be inferior.

The naive extremes both fail:
- Always exploit: get locked onto a lucky early option, never discover something better
- Always explore: waste pulls on options already known to be bad

Both UCB and Thompson Sampling solve this by being **principled** about uncertainty: explore options where you are uncertain about the true reward; concentrate on options that have demonstrated strong results.

---

## Algorithms in This Section

### Upper Confidence Bound (UCB)
**Notebook:** [01_upper_confidence_bound.ipynb](notebooks/01_upper_confidence_bound.ipynb) | **Guide:** [teaching/01_upper_confidence_bound.md](teaching/01_upper_confidence_bound.md)

UCB is a **deterministic** algorithm. At each round, it selects the option with the highest upper confidence bound:

```
UCB(i) = x̄ᵢ + √(2 × ln(n) / nᵢ)
```

- `x̄ᵢ` — current average reward from option i (exploitation term)
- `√(2 ln(n) / nᵢ)` — uncertainty bonus that grows when option i has been tried few times (exploration term)

As an option is tried more, its uncertainty bonus shrinks and it is only selected if its average reward genuinely warrants it. Options tried rarely maintain a large bonus and get periodically explored.

**Use when:** You want a theoretically grounded algorithm with provable regret bounds. UCB is easy to understand and explain: "we always try the option that could be the best, given what we know."

**Watch out for:** Requires every option to be tried at least once before the formula applies (cold-start problem). The determinism means a competitor who knows you use UCB can predict your choices.

---

### Thompson Sampling
**Notebook:** [02_thompson_sampling.ipynb](notebooks/02_thompson_sampling.ipynb) | **Guide:** [teaching/02_thompson_sampling.md](teaching/02_thompson_sampling.md)

Thompson Sampling is a **Bayesian** algorithm. It maintains a probability distribution over each option's true reward rate and samples from these distributions to decide what to try next.

For binary rewards (click / no click), it uses the Beta distribution:
- Each option starts with a uniform prior: Beta(1, 1)
- Every success increments α; every failure increments β
- At each round: sample one θᵢ from each option's Beta(αᵢ, βᵢ) distribution, then pick the option with the highest sample

Options tried rarely have wide, flat distributions — they occasionally sample very high, causing the algorithm to explore them. Options with many successes have narrow distributions concentrated near their true rate — they consistently win the sampling competition.

**Use when:** You want the best practical performance, especially early on. Thompson Sampling typically outperforms UCB empirically. The Bayesian framework lets you incorporate prior knowledge (e.g., "we expect around 5% click rate based on past campaigns").

**Watch out for:** The stochastic nature means two runs on the same data produce different decision sequences. For reporting results, run multiple simulations and average them. Only valid for the Beta-Binomial case; continuous rewards need a different conjugate model.

---

## UCB vs Thompson Sampling at a Glance

| Property | UCB | Thompson Sampling |
|----------|-----|--------------------|
| Approach | Deterministic confidence bound | Probabilistic posterior sampling |
| Theoretical regret bound | Strong, formally proven | Achieves same optimal bound |
| Empirical performance | Good | Usually better, especially early on |
| Incorporates prior knowledge | No | Yes — via initial α and β values |
| Predictable by adversaries | Yes | No (randomised policy) |

Both converge to the best option over time. The difference is speed of convergence and behaviour under uncertainty.

---

## The Dataset

`ads_ctr_optimisation.csv` — simulated click data for 10 ad creatives across 10,000 website visits. Each row is one visit; each column is one ad; the value is 1 (clicked) or 0 (not clicked). The true click-through rates differ between ads, with one clear best ad. Both algorithms must discover the best ad without knowing the true rates — they only see clicks and non-clicks as they occur.

---

## What the Teaching Guides Cover

[teaching/01_upper_confidence_bound.md](teaching/01_upper_confidence_bound.md) — derives the UCB formula from Hoeffding's inequality, explains regret bounds, step-by-step Python implementation, and the most common mistakes.

[teaching/02_thompson_sampling.md](teaching/02_thompson_sampling.md) — Bayesian inference primer, the Beta-Binomial conjugate model, how to encode prior knowledge, side-by-side comparison with UCB, and implementation with both uninformative and informative priors.
