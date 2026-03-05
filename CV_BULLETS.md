# CV bullets (In Progress)

- Building an end-to-end offline ranking evaluation pipeline (Python, MovieLens 1M) for recommender/ads ranking: leakage-safe per-user temporal split (train/val/test), candidate generation excluding seen items, and reproducible CLI workflow.
- Implemented baseline rankers (popularity and logistic regression with user/item interaction-stat features) and offline metrics (`NDCG@10`, `Recall@10`, `MAP@10`, `AUC`) to compare models under a consistent evaluation protocol.
- Added data quality checks (schema validation, deduplication, missing-value handling, dataset/split summary reports) and modular repo structure (`data/`, `models/`, `metrics/`, `scripts/`) designed for extension to CTR-style datasets (e.g., Criteo) and experimentation analysis.
