import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SPLIT_DATA_DIR,
    ensure_project_dirs,
)
from src.models.bias_mf_ranker import BiasMFImplicitRanker
from src.models.itemknn_ranker import ItemKNNRanker
from src.models.logreg_ranker import LogisticRegressionRanker
from src.models.popularity import PopularityRanker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offline ranking baselines.")
    parser.add_argument("--model", choices=["popularity", "logreg", "bias_mf", "itemknn", "all"], default="all")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=SPLIT_DATA_DIR / "ml-1m",
        help="Directory with train/val/test split CSVs",
    )
    parser.add_argument(
        "--all-interactions-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "ml-1m" / "interactions_all.csv",
        help="CSV used to get the full candidate item universe",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--movies-path",
        type=Path,
        default=RAW_DATA_DIR / "ml-1m" / "movies.dat",
        help="Movie metadata with genres for personalized features",
    )
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--mf-factors", type=int, default=32)
    parser.add_argument("--mf-epochs", type=int, default=3)
    parser.add_argument("--mf-lr", type=float, default=0.03)
    parser.add_argument("--mf-reg", type=float, default=1e-4)
    parser.add_argument("--mf-negatives-per-positive", type=int, default=3)
    parser.add_argument("--itemknn-neighbors", type=int, default=100)
    parser.add_argument("--itemknn-shrinkage", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    ensure_project_dirs()
    args = parse_args()

    train_path = args.split_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")

    if not args.all_interactions_path.exists():
        raise FileNotFoundError(f"Missing interactions file: {args.all_interactions_path}")

    train_df = pd.read_csv(train_path)
    all_interactions_df = pd.read_csv(args.all_interactions_path, usecols=["item_id"])
    all_items = set(all_interactions_df["item_id"].astype(int).unique().tolist())

    models_to_train = ["popularity", "logreg", "bias_mf", "itemknn"] if args.model == "all" else [args.model]
    if "logreg" in models_to_train and not args.movies_path.exists():
        raise FileNotFoundError(f"Missing movies metadata: {args.movies_path}")

    args.model_dir.mkdir(parents=True, exist_ok=True)

    if "popularity" in models_to_train:
        pop_model = PopularityRanker().fit(train_df)
        pop_path = args.model_dir / "popularity.joblib"
        pop_model.save(pop_path)
        print(f"Saved popularity model: {pop_path}")

    if "logreg" in models_to_train:
        logreg_model = LogisticRegressionRanker().fit(
            train_df=train_df,
            all_items=all_items,
            movies_path=args.movies_path,
            negatives_per_positive=args.negatives_per_positive,
            random_state=args.random_state,
        )
        logreg_path = args.model_dir / "logreg.joblib"
        logreg_model.save(logreg_path)
        print(f"Saved logistic regression model: {logreg_path}")

    if "bias_mf" in models_to_train:
        bias_mf_model = BiasMFImplicitRanker(
            n_factors=args.mf_factors,
            learning_rate=args.mf_lr,
            reg=args.mf_reg,
            epochs=args.mf_epochs,
            negatives_per_positive=args.mf_negatives_per_positive,
            random_state=args.random_state,
        ).fit(
            train_df=train_df,
            all_items=all_items,
        )
        bias_mf_path = args.model_dir / "bias_mf.joblib"
        bias_mf_model.save(bias_mf_path)
        print(f"Saved Bias+MF model: {bias_mf_path}")

    if "itemknn" in models_to_train:
        itemknn_model = ItemKNNRanker(
            n_neighbors=args.itemknn_neighbors,
            shrinkage=args.itemknn_shrinkage,
        ).fit(
            train_df=train_df,
            all_items=all_items,
        )
        itemknn_path = args.model_dir / "itemknn.joblib"
        itemknn_model.save(itemknn_path)
        print(f"Saved ItemKNN model: {itemknn_path}")


if __name__ == "__main__":
    main()
