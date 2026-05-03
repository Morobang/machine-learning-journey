import json

paths = [
    r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\01_regression\notebooks\03_polynomial_regression.ipynb",
    r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\01_regression\notebooks\04_support_vector_regression.ipynb",
    r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\01_regression\notebooks\05_decision_tree_regression.ipynb",
    r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\01_regression\notebooks\06_random_forest_regression.ipynb",
]
for p in paths:
    with open(p, encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=== {p.split(chr(92))[-1]} ===")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])[:120].replace("\n", " ")
            print(f"  cell {i}: {src}")
