# Categorical Encoding

Encoding strategies:
- Label Encoding: ordinal categories only
- One-Hot Encoding: nominal categories, beware high cardinality
- Target Encoding (mean encoding): can leak if not cross-validated

Practical advice:
- Use scikit-learn `OneHotEncoder` or `OrdinalEncoder` in Pipelines
- Fit encoders on training set and reuse on validation/test
- For high-cardinality, consider embedding or hashing

See notebook step: "Step Y — Encoding" for placement in the compact workflow.