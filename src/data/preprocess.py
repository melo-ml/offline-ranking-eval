import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_dirs

SCHEMA_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def load_raw_ratings(raw_dir: Path) -> pd.DataFrame:
    ratings_path = raw_dir / "ml-1m" / "ratings.dat"
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Could not find {ratings_path}. Run download step first."
        )

    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=SCHEMA_COLUMNS,
        encoding="latin-1",
    )


def validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [col for col in SCHEMA_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    for col in SCHEMA_COLUMNS:
        if not np.issubdtype(df[col].dtype, np.number):
            raise TypeError(f"Column {col} must be numeric, found {df[col].dtype}")


def preprocess_movielens(raw_dir: Path, processed_dir: Path) -> dict:
    dataset_dir = processed_dir / "ml-1m"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_ratings(raw_dir)
    validate_schema(df)

    initial_rows = len(df)

    missing_counts = {col: int(df[col].isna().sum()) for col in SCHEMA_COLUMNS}
    missing_rows = int(df[SCHEMA_COLUMNS].isna().any(axis=1).sum())
    if missing_rows > 0:
        df = df.dropna(subset=SCHEMA_COLUMNS)

    duplicate_mask = df.duplicated(subset=SCHEMA_COLUMNS, keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    if duplicates_removed > 0:
        df = df.loc[~duplicate_mask].copy()

    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    df["label"] = (df["rating"] >= 4.0).astype(int)

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df_pos = df[df["label"] == 1].copy()

    all_path = dataset_dir / "interactions_all.csv"
    pos_path = dataset_dir / "interactions_positive.csv"
    report_path = dataset_dir / "dq_report.json"

    df.to_csv(all_path, index=False)
    df_pos.to_csv(pos_path, index=False)

    report = {
        "initial_rows": int(initial_rows),
        "final_rows": int(len(df)),
        "positive_rows": int(len(df_pos)),
        "duplicates_removed": int(duplicates_removed),
        "missing_rows_removed": int(missing_rows),
        "missing_counts": missing_counts,
        "num_users": int(df["user_id"].nunique()),
        "num_items": int(df["item_id"].nunique()),
        "positive_rate": float(df["label"].mean()),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved all interactions: {all_path}")
    print(f"Saved positive interactions: {pos_path}")
    print(f"Saved DQ report: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens data.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    preprocess_movielens(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
