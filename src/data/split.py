import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, SPLIT_DATA_DIR, ensure_project_dirs


SPLIT_COLUMNS = ["user_id", "item_id", "rating", "timestamp", "label"]


def make_leave_last_two_split(
    positive_df: pd.DataFrame,
    min_interactions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_parts = []
    val_rows = []
    test_rows = []
    dropped_users = 0

    # Use stable sort so equal timestamps keep deterministic original order.
    grouped = positive_df.sort_values(["user_id", "timestamp"], kind="mergesort").groupby(
        "user_id", sort=False
    )

    for user_id, user_df in grouped:
        if len(user_df) < min_interactions:
            dropped_users += 1
            continue

        train_parts.append(user_df.iloc[:-2])
        val_rows.append(user_df.iloc[-2])
        test_rows.append(user_df.iloc[-1])

    if train_parts:
        train_df = pd.concat(train_parts, ignore_index=True)
    else:
        train_df = pd.DataFrame(columns=SPLIT_COLUMNS)

    val_df = pd.DataFrame(val_rows).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).reset_index(drop=True)

    tied_train_val_users = 0
    tied_val_test_users = 0

    if not train_df.empty and not val_df.empty and not test_df.empty:
        train_max = train_df.groupby("user_id")["timestamp"].max().rename("train_ts")
        val_ts = val_df.set_index("user_id")["timestamp"].rename("val_ts")
        test_ts = test_df.set_index("user_id")["timestamp"].rename("test_ts")
        merged = train_max.to_frame().join(val_ts, how="inner").join(test_ts, how="inner")

        # Equal timestamps are allowed (multiple interactions in the same second).
        # We only require non-decreasing temporal order across train/val/test.
        leakage_free = ((merged["train_ts"] <= merged["val_ts"]) & (merged["val_ts"] <= merged["test_ts"]))
        if not leakage_free.all():
            raise ValueError("Temporal leakage check failed. Expected train <= val <= test per user.")

        tied_train_val_users = int((merged["train_ts"] == merged["val_ts"]).sum())
        tied_val_test_users = int((merged["val_ts"] == merged["test_ts"]).sum())

    summary = {
        "total_positive_rows": int(len(positive_df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_users": int(train_df["user_id"].nunique()) if not train_df.empty else 0,
        "val_users": int(val_df["user_id"].nunique()) if not val_df.empty else 0,
        "test_users": int(test_df["user_id"].nunique()) if not test_df.empty else 0,
        "dropped_users_lt_min_interactions": int(dropped_users),
        "min_interactions_required": int(min_interactions),
        "users_with_train_eq_val_timestamp": tied_train_val_users,
        "users_with_val_eq_test_timestamp": tied_val_test_users,
    }

    return train_df, val_df, test_df, summary


def run_split(
    positive_path: Path,
    split_dir: Path,
    min_interactions: int = 3,
) -> dict:
    split_dataset_dir = split_dir / "ml-1m"
    split_dataset_dir.mkdir(parents=True, exist_ok=True)

    df_pos = pd.read_csv(positive_path)
    train_df, val_df, test_df, summary = make_leave_last_two_split(
        df_pos,
        min_interactions=min_interactions,
    )

    train_path = split_dataset_dir / "train.csv"
    val_path = split_dataset_dir / "val.csv"
    test_path = split_dataset_dir / "test.csv"
    summary_path = split_dataset_dir / "split_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved train split: {train_path}")
    print(f"Saved val split: {val_path}")
    print(f"Saved test split: {test_path}")
    print(f"Saved split summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leave-last splits per user.")
    parser.add_argument(
        "--positive-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "ml-1m" / "interactions_positive.csv",
    )
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DATA_DIR)
    parser.add_argument("--min-interactions", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    run_split(
        positive_path=args.positive_path,
        split_dir=args.split_dir,
        min_interactions=args.min_interactions,
    )


if __name__ == "__main__":
    main()
