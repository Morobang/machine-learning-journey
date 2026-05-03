import json, sys

sys.stdout.reconfigure(encoding="utf-8")

paths = [
    r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\01_logistic_regression.ipynb",
    r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\04_kernel_svm.ipynb",
    r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\05_naive_bayes.ipynb",
    r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\06_decision_tree_classification.ipynb",
    r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\07_random_forest_classification.ipynb",
]
for p in paths:
    with open(p, encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=== {p.split(chr(92))[-1]} ===")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])[:120].replace("\n", " ")
            print(f"  cell {i}: {src}")
