import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\03_reinforcement_learning\notebooks\02_thompson_sampling.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title / intro
cells[0]["source"] = [
    "# Thompson Sampling\n",
    "\n",
    "## The Same Problem, A Different Philosophy\n",
    "\n",
    "We are solving the same **multi-armed bandit problem** from the UCB notebook: 10 ads, 10,000 users, unknown click-through rates. We want to find the best ad while wasting as little budget as possible on bad ones.\n",
    "\n",
    "UCB approached this deterministically: compute a formula, pick the highest number. **Thompson Sampling** takes a fundamentally different, probabilistic approach:\n",
    "\n",
    "> Instead of computing a confidence bound, maintain a **probability distribution** over each ad's true click rate, and sample from it to make each decision.\n",
    "\n",
    "---\n",
    "\n",
    "## The Bayesian Intuition\n",
    "\n",
    "Thompson Sampling is a **Bayesian algorithm**. Here is the core idea:\n",
    "\n",
    "1. Start with a **prior belief** that each ad has an unknown click rate somewhere between 0 and 1 (uniform distribution — all equally plausible).\n",
    "2. Each time you show an ad and observe a click (reward = 1) or no-click (reward = 0), **update your belief** about that ad.\n",
    "3. When choosing which ad to show next, **sample a number** from each ad's current belief distribution. Show the ad with the highest sampled value.\n",
    "4. Repeat.\n",
    "\n",
    "Over time, the belief distributions for bad ads tighten around low values and the best ad's distribution tightens around its true click rate — making it almost always win the sampling step.\n",
    "\n",
    "---\n",
    "\n",
    "## The Beta Distribution: Perfect for Click Rates\n",
    "\n",
    "Click rates are probabilities between 0 and 1. The **Beta distribution** is the natural choice for modelling a probability:\n",
    "\n",
    "$$\\text{Beta}(\\alpha, \\beta)$$\n",
    "\n",
    "| Parameter | Meaning in our context |\n",
    "|-----------|------------------------|\n",
    "| $\\alpha$ | Number of clicks observed (successes) + 1 |\n",
    "| $\\beta$ | Number of non-clicks observed (failures) + 1 |\n",
    "\n",
    "Adding 1 to both is the **Laplace smoothing** that gives the uniform prior (Beta(1,1)) before any data. As clicks accumulate, $\\alpha$ grows and the distribution shifts right (towards higher click rates). As no-clicks accumulate, $\\beta$ grows and it shifts left.\n",
    "\n",
    "---\n",
    "\n",
    "## UCB vs Thompson Sampling\n",
    "\n",
    "| | UCB | Thompson Sampling |\n",
    "|-|-----|-------------------|\n",
    "| **Approach** | Deterministic formula | Probabilistic sampling |\n",
    "| **Exploration** | Explicit confidence bonus | Implicit through sampling variance |\n",
    "| **Computation** | Simpler math | Requires sampling from Beta distribution |\n",
    "| **Empirical performance** | Strong | Often better in practice |\n",
    "| **Theoretical guarantees** | Logarithmic regret | Bayesian optimal (Lai-Robbins bound) |\n",
    "\n",
    "**In practice**, Thompson Sampling often converges faster and is more robust because natural uncertainty in the distribution handles exploration automatically — no tuning needed.\n",
    "\n",
    "---\n",
    "\n",
    "## What We Will Build\n",
    "\n",
    "1. Maintain Beta distribution parameters for all 10 ads\n",
    "2. At each round: sample from each ad's Beta distribution, show the ad with the highest sample\n",
    "3. Update the distribution based on the observed reward\n",
    "4. Visualise which ad the algorithm converges on after 10,000 rounds"
]

# Cell 1 — libraries
cells[1]["source"] = [
    "## Step 1: Import Libraries\n",
    "\n",
    "| Library | Why we need it |\n",
    "|---------|---------------|\n",
    "| `numpy` | Array operations |\n",
    "| `matplotlib` | Visualising the selection histogram |\n",
    "| `pandas` | Loading the CSV dataset |\n",
    "| `random` | `random.betavariate(alpha, beta)` — samples from the Beta distribution |"
]

# Cell 3 — dataset
cells[3]["source"] = [
    "## Step 2: Load the Dataset\n",
    "\n",
    "Same dataset as the UCB notebook: `ads_ctr_optimization.csv` with 10,000 rows and 10 columns.\n",
    "\n",
    "Each row is a user. Each column is one ad. The value (0 or 1) indicates whether that user *would have* clicked that ad. In a real system you only observe the reward for the ad you chose — the rest are counterfactual.\n",
    "\n",
    "**This is a simulated environment**, which is standard practice for testing bandit algorithms before deploying them."
]

# Cell 5 — implementing Thompson Sampling
cells[5]["source"] = [
    "## Step 3: Implement Thompson Sampling\n",
    "\n",
    "We track two counters per ad:\n",
    "\n",
    "| Variable | Meaning |\n",
    "|----------|---------|\n",
    "| `numbers_of_rewards_1[i]` | How many times ad $i$ was clicked ($\\alpha - 1$) |\n",
    "| `numbers_of_rewards_0[i]` | How many times ad $i$ was not clicked ($\\beta - 1$) |\n",
    "\n",
    "**The algorithm at each round:**\n",
    "\n",
    "1. For each ad $i$, draw a **random sample** from `Beta(rewards_1[i] + 1, rewards_0[i] + 1)`\n",
    "   - Early on: both parameters are 1, so Beta(1,1) is uniform — any click rate is equally likely\n",
    "   - After 50 clicks and 10 non-clicks: Beta(51, 11) peaks near 0.82 — strong evidence of a good ad\n",
    "2. Select the ad with the **highest sampled value** — this is stochastic, not deterministic\n",
    "3. Observe the reward\n",
    "4. Increment `rewards_1` if clicked, `rewards_0` if not\n",
    "\n",
    "**Why does sampling work for exploration?** An ad with only 3 observations has high variance in its Beta distribution — it can easily sample a high value by chance, giving it another trial. An ad with 500 observations has a tight distribution — it only wins the sampling step if its true rate is genuinely high. Exploration is automatic and diminishes naturally as evidence accumulates."
]

# Cell 7 — visualising
cells[7]["source"] = [
    "## Step 4: Visualise the Results\n",
    "\n",
    "The histogram shows how many of the 10,000 rounds each ad was selected.\n",
    "\n",
    "**What you should observe:**\n",
    "\n",
    "- **One bar dominates** — the ad with the highest true click rate gets selected the vast majority of the time\n",
    "- **All other bars are short but present** — Thompson Sampling explored all ads early, then committed to the best\n",
    "- The convergence should be even sharper than UCB, because sampling variance decreases faster for well-tested ads\n",
    "\n",
    "**Comparing to UCB:** Both algorithms converge to the same best ad. Thompson Sampling tends to waste fewer rounds on bad ads because it naturally gives them less exploration budget once evidence against them accumulates.\n",
    "\n",
    "**Real-world deployment note:** Thompson Sampling is used in production at companies including Google, Microsoft, and LinkedIn for personalised recommendations and ad selection — precisely because it handles the exploration-exploitation tradeoff automatically without manual tuning of exploration parameters."
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Thompson Sampling notebook updated successfully.")
