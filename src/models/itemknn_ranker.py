from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ItemKNNRanker:
    """
    Item-based KNN collaborative filtering for implicit feedback.
    Similarity: cosine with optional shrinkage.
    Score(u, i): sum of similarities between item i and user's historical items.
    """

    def __init__(self, n_neighbors: int = 100, shrinkage: float = 10.0) -> None:
        self.n_neighbors = int(n_neighbors)
        self.shrinkage = float(shrinkage)

        self.item_id_to_idx: dict[int, int] = {}
        self.user_histories: dict[int, np.ndarray] = {}
        self.neighbor_indices: list[np.ndarray] = []
        self.neighbor_sims: list[np.ndarray] = []

    def fit(self, train_df: pd.DataFrame, all_items: set[int]) -> "ItemKNNRanker":
        user_ids = sorted(train_df["user_id"].astype(int).unique().tolist())
        item_ids = sorted(int(x) for x in all_items)

        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

        n_users = len(user_ids)
        n_items = len(item_ids)

        # Build binary user-item interaction matrix.
        r = np.zeros((n_users, n_items), dtype=np.float32)
        for row in train_df.itertuples(index=False):
            u_idx = user_id_to_idx.get(int(row.user_id))
            i_idx = self.item_id_to_idx.get(int(row.item_id))
            if u_idx is not None and i_idx is not None:
                r[u_idx, i_idx] = 1.0

        for uid, group in train_df.groupby("user_id")["item_id"]:
            hist = [self.item_id_to_idx[int(iid)] for iid in group.astype(int).tolist() if int(iid) in self.item_id_to_idx]
            self.user_histories[int(uid)] = np.array(hist, dtype=np.int32)

        cooc = r.T @ r
        item_norms = np.sqrt(np.maximum(np.diag(cooc), 1e-12))
        denom = item_norms[:, None] * item_norms[None, :]
        sim = np.divide(cooc, denom, out=np.zeros_like(cooc), where=denom > 0)

        if self.shrinkage > 0:
            sim *= cooc / (cooc + self.shrinkage)

        np.fill_diagonal(sim, 0.0)

        self.neighbor_indices = []
        self.neighbor_sims = []
        k = min(self.n_neighbors, n_items - 1) if n_items > 1 else 0

        if k <= 0:
            self.neighbor_indices = [np.array([], dtype=np.int32) for _ in range(n_items)]
            self.neighbor_sims = [np.array([], dtype=np.float32) for _ in range(n_items)]
            return self

        for i_idx in range(n_items):
            row = sim[i_idx]
            top_idx = np.argpartition(row, -k)[-k:]
            top_scores = row[top_idx]
            pos_mask = top_scores > 0
            if not np.any(pos_mask):
                self.neighbor_indices.append(np.array([], dtype=np.int32))
                self.neighbor_sims.append(np.array([], dtype=np.float32))
                continue

            top_idx = top_idx[pos_mask]
            top_scores = top_scores[pos_mask]
            order = np.argsort(top_scores)[::-1]
            self.neighbor_indices.append(top_idx[order].astype(np.int32))
            self.neighbor_sims.append(top_scores[order].astype(np.float32))

        return self

    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        n_items = len(self.item_id_to_idx)
        if n_items == 0:
            return np.zeros(item_ids.shape[0], dtype=float)

        history = self.user_histories.get(int(user_id))
        if history is None or history.size == 0:
            return np.zeros(item_ids.shape[0], dtype=float)

        full_scores = np.zeros(n_items, dtype=np.float32)
        for seen_i in history:
            nbr_idx = self.neighbor_indices[int(seen_i)]
            nbr_sim = self.neighbor_sims[int(seen_i)]
            if nbr_idx.size > 0:
                full_scores[nbr_idx] += nbr_sim

        query_idx = np.array([self.item_id_to_idx.get(int(iid), -1) for iid in item_ids], dtype=np.int32)
        result = np.zeros(item_ids.shape[0], dtype=float)
        known_mask = query_idx >= 0
        if np.any(known_mask):
            result[known_mask] = full_scores[query_idx[known_mask]]
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ItemKNNRanker":
        return joblib.load(path)
