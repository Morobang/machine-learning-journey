import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\03_reinforcement_learning\notebooks\01_upper_confidence_bound.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title / intro
cells[0]["source"] = [
    "# Upper Confidence Bound (UCB)\n",
    "\n",
    "## The Problem: Where Should You Show Your Ad?\n",
    "\n",
    "Imagine you are running a digital advertising campaign. You have **10 different ad designs** and you want to show users the one that gets clicked the most — but you do not know which ad is best at the start. You have two competing pressures:\n",
    "\n",
    "- **Exploitation:** Show the ad that has worked best *so far*.\n",
    "- **Exploration:** Try ads you have not shown much, in case one of them is actually much better.\n",
    "\n",
    "This tension is called the **Exploration vs Exploitation dilemma**, and it appears in almost every sequential decision-making problem: recommendation systems, clinical trials, stock trading, and robot navigation.\n",
    "\n",
    "---\n",
    "\n",
    "## The Multi-Armed Bandit Framework\n",
    "\n",
    "The classic framing is the **multi-armed bandit problem**:\n",
    "\n",
    "> You are in a casino with 10 slot machines (\"one-armed bandits\"). Each machine pays out at a fixed but unknown probability. You have 10,000 pulls total. How do you maximise your total winnings?\n",
    "\n",
    "In our case:\n",
    "- Each **slot machine** = one ad design\n",
    "- A **pull** = showing the ad to one user\n",
    "- A **reward of 1** = the user clicked\n",
    "- A **reward of 0** = the user did not click\n",
    "\n",
    "---\n",
    "\n",
    "## Why Not Just Test Each Ad Equally?\n",
    "\n",
    "Pure **random exploration** wastes budget: you keep testing ads that are clearly bad. Pure **greedy exploitation** of the current best ad can lock you into a suboptimal choice early on.\n",
    "\n",
    "UCB solves this by building a **confidence interval** around each ad's estimated click-through rate. Ads with fewer observations get a wider (higher) confidence bound — which gives them a chance to be selected and tested. As an ad gets selected more, its bound tightens to its true rate. You always pick the ad with the **highest upper confidence bound**.\n",
    "\n",
    "---\n",
    "\n",
    "## The UCB Formula\n",
    "\n",
    "At round $n$, for ad $i$:\n",
    "\n",
    "$$\\text{UCB}_i(n) = \\bar{r}_i + \\sqrt{\\frac{3}{2} \\cdot \\frac{\\ln(n)}{N_i(n)}}$$\n",
    "\n",
    "| Symbol | Meaning |\n",
    "|--------|---------|\n",
    "| $\\bar{r}_i$ | Average reward (click rate) for ad $i$ so far |\n",
    "| $n$ | Total number of rounds played |\n",
    "| $N_i(n)$ | Number of times ad $i$ has been selected |\n",
    "| $\\ln(n)$ | Natural log of the total rounds — grows slowly, so the bonus shrinks over time |\n",
    "\n",
    "**The bonus term** $\\sqrt{\\frac{3}{2} \\cdot \\frac{\\ln(n)}{N_i(n)}}$ is large when ad $i$ has been selected few times and shrinks as $N_i$ grows. This naturally balances exploration and exploitation.\n",
    "\n",
    "---\n",
    "\n",
    "## What We Will Build\n",
    "\n",
    "1. Simulate 10,000 users seeing ads from a dataset of 10 options\n",
    "2. Use UCB to decide which ad to show at each step\n",
    "3. Visualise which ad the algorithm converges on — and why that matters for a real campaign"
]

# Cell 1 — importing libraries
cells[1]["source"] = [
    "## Step 1: Import Libraries\n",
    "\n",
    "| Library | Why we need it |\n",
    "|---------|---------------|\n",
    "| `numpy` | Array operations and mathematical functions |\n",
    "| `matplotlib` | Plotting the histogram of ad selections |\n",
    "| `pandas` | Loading the dataset from CSV |\n",
    "| `math` | `math.log()` and `math.sqrt()` for the UCB formula |"
]

# Cell 3 — importing dataset
cells[3]["source"] = [
    "## Step 2: Load the Dataset\n",
    "\n",
    "The dataset `ads_ctr_optimization.csv` contains **10,000 rows and 10 columns**.\n",
    "\n",
    "- Each **row** represents one user visiting the site.\n",
    "- Each **column** represents one ad design (Ad 1 through Ad 10).\n",
    "- A value of **1** means: *if this user had been shown this ad, they would have clicked.*\n",
    "- A value of **0** means: *they would not have clicked.*\n",
    "\n",
    "**Important:** In a real deployment, you only observe the reward for the one ad you actually showed. The other columns are hidden from you — the dataset is a simulation that lets us verify our algorithm found the best ad.\n",
    "\n",
    "The task: using only the feedback column for the ad we select each round, find the best-performing ad as efficiently as possible."
]

# Cell 5 — implementing UCB
cells[5]["source"] = [
    "## Step 3: Implement the UCB Algorithm\n",
    "\n",
    "This is the core of the notebook. Before reading the code, understand what we are tracking:\n",
    "\n",
    "| Variable | Type | Purpose |\n",
    "|----------|------|---------|\n",
    "| `ads_selected` | list | History of which ad was shown at each round |\n",
    "| `numbers_of_selections` | list of 10 ints | How many times each ad has been shown so far |\n",
    "| `sums_of_rewards` | list of 10 ints | Total clicks accumulated for each ad |\n",
    "| `total_reward` | int | Running total of all clicks — our performance metric |\n",
    "\n",
    "**The algorithm at each round $n$:**\n",
    "\n",
    "1. For each ad $i$, compute its UCB:\n",
    "   - If it has been shown before: `average_reward + delta_i` (the UCB formula)\n",
    "   - If it has **never** been shown: set the upper bound to infinity (`1e400`) — force the algorithm to try every ad at least once\n",
    "2. Select the ad with the **highest upper bound**\n",
    "3. Observe the reward (did this user click?)\n",
    "4. Update `numbers_of_selections` and `sums_of_rewards` for the chosen ad\n",
    "\n",
    "After all 10,000 rounds, `ads_selected` tells the complete story of which ad the algorithm preferred at each step."
]

# Cell 7 — visualising results
cells[7]["source"] = [
    "## Step 4: Visualise the Results\n",
    "\n",
    "The histogram shows **how many times each ad was selected** over 10,000 rounds.\n",
    "\n",
    "**What to look for:**\n",
    "\n",
    "- One ad should have a dramatically **taller bar** than all the others — this is the ad UCB identified as best.\n",
    "- The other bars are short but non-zero — UCB explored all 10 ads early on, then committed to the winner.\n",
    "- Compare the tallest bar to a uniform baseline of 1,000 (10,000 rounds / 10 ads). The more the algorithm converges, the taller the winner bar and the shorter the rest.\n",
    "\n",
    "**Regret:** The ads that got many selections but were not the best represent *regret* — rounds where we could have shown the better ad. A good bandit algorithm minimises cumulative regret. UCB is proven to achieve **logarithmic regret** over time, meaning the per-round regret shrinks as the algorithm gets more confident.\n",
    "\n",
    "**Why this matters for a real campaign:** In a 10,000-user campaign, the difference between showing a 2% CTR ad vs a 5% CTR ad is 300 extra clicks — or potentially hundreds of conversions and thousands in revenue. UCB finds the best ad *while running the campaign*, not after a separate A/B test."
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("UCB notebook updated successfully.")
