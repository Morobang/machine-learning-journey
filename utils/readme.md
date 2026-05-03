# utils/

This folder is superseded by [`src/`](../src/).

All reusable ML utilities — preprocessing helpers, evaluation reports, and EDA plots — now live in the `src/` library, which is properly structured, tested, and importable from any notebook.

## Importing src/ in a notebook

Add this at the top of any notebook, adjusting the `parents` depth to match the notebook's location:

```python
import sys
from pathlib import Path

# Adjust depth: 2 for section/notebooks/, 3 for projects/tier/name/notebooks/
sys.path.insert(0, str(Path().resolve().parents[2]))

from src.preprocessing import split_features, scale_features
from src.evaluation import regression_report, plot_confusion_matrix
from src.visualization import plot_distributions, plot_correlation_heatmap
```

## Available modules

| Import path | What it provides |
|---|---|
| `src.preprocessing` | `split_features`, `scale_features`, `encode_labels`, `one_hot_encode`, `impute_median` |
| `src.evaluation` | `regression_report`, `classification_report_full`, `cross_val_report` |
| `src.evaluation` | `plot_actual_vs_predicted`, `plot_residuals`, `plot_confusion_matrix`, `plot_roc_curve`, `plot_feature_importance`, `plot_learning_curve` |
| `src.visualization` | `plot_distributions`, `plot_correlation_heatmap`, `plot_missing_values`, `plot_class_balance`, `plot_pairplot` |

See [`src/`](../src/) for full source code and [`tests/`](../tests/) for usage examples.