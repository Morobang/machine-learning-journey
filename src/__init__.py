"""
src — reusable utilities for the machine-learning-journey notebooks.

Import pattern in notebooks:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path().resolve().parents[2]))
    from src.preprocessing import split_features, scale_features
    from src.evaluation import regression_report, classification_report_full
    from src.visualization import plot_regression, plot_confusion_matrix
"""
