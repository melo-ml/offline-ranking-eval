import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
    SPLIT_DATA_DIR,
    ensure_project_dirs,
)
from src.metrics.ranking import (
    auc_from_positive_score,
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.models.bias_mf_ranker import BiasMFImplicitRanker
from src.models.itemknn_ranker import ItemKNNRanker
from src.models.logreg_ranker import LogisticRegressionRanker
from src.models.popularity import PopularityRanker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranking models offline.")
    parser.add_argument("--model", choices=["popularity", "logreg", "bias_mf", "itemknn"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DATA_DIR / "ml-1m")
    parser.add_argument(
        "--all-interactions-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "ml-1m" / "interactions_all.csv",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def load_model(model_name: str, model_dir: Path):
    model_path = model_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_name == "popularity":
        return PopularityRanker.load(model_path)
    if model_name == "logreg":
        return LogisticRegressionRanker.load(model_path)
    if model_name == "bias_mf":
        return BiasMFImplicitRanker.load(model_path)
    if model_name == "itemknn":
        return ItemKNNRanker.load(model_path)

    raise ValueError(f"Unsupported model: {model_name}")


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.array([], dtype=int)

    top_n = min(k, scores.size)
    part = np.argpartition(scores, -top_n)[-top_n:]
    return part[np.argsort(scores[part])[::-1]]


def main() -> None:
    ensure_project_dirs()
    args = parse_args()

    train_df = pd.read_csv(args.split_dir / "train.csv", usecols=["user_id", "item_id"])
    val_df = pd.read_csv(args.split_dir / "val.csv", usecols=["user_id", "item_id"])
    test_df = pd.read_csv(args.split_dir / "test.csv", usecols=["user_id", "item_id"])

    if args.split == "val":
        target_df = val_df
        seen_df = train_df
    else:
        target_df = test_df
        seen_df = pd.concat([train_df, val_df], ignore_index=True)

    model = load_model(args.model, args.model_dir)
    all_items = pd.read_csv(args.all_interactions_path, usecols=["item_id"])["item_id"].astype(int).unique()
    all_items = np.sort(all_items)

    seen_by_user = seen_df.groupby("user_id")["item_id"].apply(set).to_dict()
    user_targets = target_df.groupby("user_id")["item_id"].apply(list).to_dict()

    users = sorted(user_targets.keys())
    if args.max_users is not None:
        users = users[: args.max_users]

    recalls = []
    ndcgs = []
    maps = []
    aucs = []

    for user_id in tqdm(users, desc=f"Evaluating {args.model} on {args.split}"):
        relevant_items = set(int(x) for x in user_targets[user_id])
        seen_items = seen_by_user.get(user_id, set())

        if seen_items:
            candidate_items = all_items[~np.isin(all_items, np.fromiter(seen_items, dtype=np.int64), assume_unique=False)]
        else:
            candidate_items = all_items

        if candidate_items.size == 0:
            continue

        scores = model.score_items(user_id=int(user_id), item_ids=candidate_items)
        if scores.size == 0:
            continue

        top_idx = top_k_indices(scores=scores, k=args.k)
        ranked_items = candidate_items[top_idx].astype(int).tolist()

        recalls.append(recall_at_k(ranked_items, relevant_items, args.k))
        ndcgs.append(ndcg_at_k(ranked_items, relevant_items, args.k))
        maps.append(average_precision_at_k(ranked_items, relevant_items, args.k))

        positive_mask = np.isin(candidate_items, np.fromiter(relevant_items, dtype=np.int64), assume_unique=False)
        if positive_mask.any() and (~positive_mask).any():
            positive_scores = scores[positive_mask]
            negative_scores = scores[~positive_mask]
            auc_values = [auc_from_positive_score(float(pos), negative_scores) for pos in positive_scores]
            aucs.append(float(np.mean(auc_values)))

    report = {
        "model": args.model,
        "split": args.split,
        "k": int(args.k),
        "num_users_evaluated": int(len(recalls)),
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "map_at_k": float(np.mean(maps)) if maps else 0.0,
        "auc": float(np.mean(aucs)) if aucs else None,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"{args.model}_{args.split}_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
