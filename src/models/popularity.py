from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class PopularityRanker:
    def __init__(self) -> None:
        self.item_scores: dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "PopularityRanker":
        counts = train_df["item_id"].value_counts()
        self.item_scores = {int(item_id): float(cnt) for item_id, cnt in counts.items()}
        return self

    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        del user_id
        return np.array([self.item_scores.get(int(item_id), 0.0) for item_id in item_ids], dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PopularityRanker":
        return joblib.load(path)
