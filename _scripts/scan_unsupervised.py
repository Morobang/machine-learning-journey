import json, sys
sys.stdout.reconfigure(encoding="utf-8")

paths = [
    r"c:\Users\User\Documents\Github\machine-learning-journey\02_unsupervised_learning\clustering\notebooks\01_k_means_clustering.ipynb",
    r"c:\Users\User\Documents\Github\machine-learning-journey\02_unsupervised_learning\association_rules\notebooks\01_apriori.ipynb",
    r"c:\Users\User\Documents\Github\machine-learning-journey\02_unsupervised_learning\association_rules\notebooks\02_eclat.ipynb",
]
for p in paths:
    with open(p, encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=== {p.split(chr(92))[-1]} ({len(nb['cells'])} cells) ===")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])[:150].replace("\n", " ")
            print(f"  cell {i}: {src}")
