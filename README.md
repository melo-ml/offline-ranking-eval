# Offline Ranking Evaluation (MovieLens 1M)

Industry-style, reproducible offline ranking evaluation project for recommender/ads ranking and growth experimentation DS interviews.

## MVP scope (Milestone 1-2)
- Dataset: MovieLens 1M (implicit label from explicit ratings)
- Label definition: `rating >= 4 -> positive (1)`, else `0`
- Leakage-safe split: per-user leave-last-1 (test) and second-last-1 (val), training on older positives only
- Baselines:
  - `PopularityRanker`
  - `LogisticRegressionRanker` with user/item interaction-count + genre-match features
  - `BiasMFImplicitRanker` (user bias + item bias + latent factors)
  - `ItemKNNRanker` (item-item cosine similarity + top-K neighbors)
- Metrics: `NDCG@10`, `Recall@10`, `MAP@10`, `AUC`
- Evaluation: per-user ranking over candidate items excluding seen items
- Data quality checks: schema validation, missing-value checks, dedup, summary report

## Repo tree
```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── artifacts/
│   ├── models/
│   └── reports/
└── src/
    ├── __init__.py
    ├── config.py
    ├── data/
    │   ├── __init__.py
    │   ├── download_movielens.py
    │   ├── preprocess.py
    │   └── split.py
    ├── metrics/
    │   ├── __init__.py
    │   └── ranking.py
    ├── models/
    │   ├── __init__.py
    │   ├── bias_mf_ranker.py
    │   ├── itemknn_ranker.py
    │   ├── logreg_ranker.py
    │   └── popularity.py
    ├── scripts/
    │   ├── __init__.py
    │   ├── download_data.py
    │   ├── preprocess_data.py
    │   ├── make_splits.py
    │   ├── train.py
    │   └── evaluate.py
    └── utils/
        └── __init__.py
```

## Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step-by-step run
### 1) Download MovieLens 1M
```bash
python -m src.scripts.download_data
```
Expected raw file after extraction:
- `data/raw/ml-1m/ratings.dat`

### 2) Preprocess + data quality checks
```bash
python -m src.scripts.preprocess_data
```
Outputs:
- `data/processed/ml-1m/interactions_all.csv`
- `data/processed/ml-1m/interactions_positive.csv`
- `data/processed/ml-1m/dq_report.json`

### 3) Build leakage-safe temporal split
```bash
python -m src.scripts.make_splits --min-interactions 3
```
Outputs:
- `data/splits/ml-1m/train.csv`
- `data/splits/ml-1m/val.csv`
- `data/splits/ml-1m/test.csv`
- `data/splits/ml-1m/split_summary.json`

### 4) Train baselines
Train all baselines:
```bash
python -m src.scripts.train --model all --negatives-per-positive 3 \
  --mf-factors 32 --mf-epochs 3 --mf-negatives-per-positive 3 \
  --itemknn-neighbors 100 --itemknn-shrinkage 10
```
Or train one:
```bash
python -m src.scripts.train --model popularity
python -m src.scripts.train --model logreg --negatives-per-positive 3
python -m src.scripts.train --model bias_mf --mf-factors 32 --mf-epochs 3 --mf-negatives-per-positive 3
python -m src.scripts.train --model itemknn --itemknn-neighbors 100 --itemknn-shrinkage 10
```
Outputs:
- `artifacts/models/popularity.joblib`
- `artifacts/models/logreg.joblib`
- `artifacts/models/bias_mf.joblib`
- `artifacts/models/itemknn.joblib`

### 5) Evaluate
Evaluate on validation:
```bash
python -m src.scripts.evaluate --model popularity --split val --k 10
python -m src.scripts.evaluate --model logreg --split val --k 10
python -m src.scripts.evaluate --model bias_mf --split val --k 10
python -m src.scripts.evaluate --model itemknn --split val --k 10
```
Evaluate on test:
```bash
python -m src.scripts.evaluate --model popularity --split test --k 10
python -m src.scripts.evaluate --model logreg --split test --k 10
python -m src.scripts.evaluate --model bias_mf --split test --k 10
python -m src.scripts.evaluate --model itemknn --split test --k 10
```
Optional fast smoke run on subset of users:
```bash
python -m src.scripts.evaluate --model bias_mf --split test --k 10 --max-users 500
```
Outputs:
- `artifacts/reports/<model>_<split>_metrics.json`

## Example Results (Test@10, local run)
| Model (Test@10) | Recall@10 | NDCG@10 | MAP@10 | AUC |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0409 | 0.0192 | 0.0128 | 0.8274 |
| LogReg (count + genre) | 0.0408 | 0.0197 | 0.0133 | 0.8320 |
| Bias-MF (f=64, e=10) | 0.0729 | 0.0361 | 0.0251 | 0.9058 |
| ItemKNN | 0.0592 | 0.0292 | 0.0202 | 0.8308 |

## Notes for extension (Milestone 3+)
- Add `XGBoost` or stronger MF baseline (BPR, LightFM, neural CF)
- Add richer user/item/context features
- Add Criteo-like CTR dataset adapter with shared evaluator API
- Add experiment tracking and confidence intervals / bootstrap for metric deltas

## Reproducibility tips
- Keep `--random-state` fixed during training
- Keep split policy fixed (`leave-last-1`, `last-1`) for fair comparison
- Log dataset snapshot sizes from `dq_report.json` and `split_summary.json`
