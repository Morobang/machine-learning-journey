import json, sys
sys.stdout.reconfigure(encoding="utf-8")

path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\01_logistic_regression.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown":
        src = "".join(c["source"])[:200].replace("\n", " ")
        print(f"cell {i}: {src}")
