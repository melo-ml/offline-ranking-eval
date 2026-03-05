from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BiasMFImplicitRanker:
    """
    Implicit-feedback matrix factorization with user/item bias terms.
    Optimized with SGD on sampled positives and negatives using logistic loss.
    """

    def __init__(
        self,
        n_factors: int = 32,
        learning_rate: float = 0.03,
        reg: float = 1e-4,
        epochs: int = 3,
        negatives_per_positive: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_factors = int(n_factors)
        self.learning_rate = float(learning_rate)
        self.reg = float(reg)
        self.epochs = int(epochs)
        self.negatives_per_positive = int(negatives_per_positive)
        self.random_state = int(random_state)

        self.global_bias: float = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

        self.user_id_to_idx: dict[int, int] = {}
        self.item_id_to_idx: dict[int, int] = {}

    @staticmethod
    def _sigmoid(x: float) -> float:
        x_clipped = np.clip(x, -35.0, 35.0)
        return float(1.0 / (1.0 + np.exp(-x_clipped)))

    def _check_fitted(self) -> None:
        if (
            self.user_bias is None
            or self.item_bias is None
            or self.user_factors is None
            or self.item_factors is None
        ):
            raise ValueError("Model is not fitted.")

    def _sgd_update(self, u_idx: int, i_idx: int, label: float) -> None:
        self._check_fitted()
        assert self.user_bias is not None
        assert self.item_bias is not None
        assert self.user_factors is not None
        assert self.item_factors is not None

        u_vec = self.user_factors[u_idx].copy()
        i_vec = self.item_factors[i_idx].copy()

        score = self.global_bias + self.user_bias[u_idx] + self.item_bias[i_idx] + float(np.dot(u_vec, i_vec))
        pred = self._sigmoid(score)
        err = label - pred

        lr = self.learning_rate
        reg = self.reg

        self.global_bias += lr * err
        self.user_bias[u_idx] += lr * (err - reg * self.user_bias[u_idx])
        self.item_bias[i_idx] += lr * (err - reg * self.item_bias[i_idx])

        self.user_factors[u_idx] += lr * (err * i_vec - reg * u_vec)
        self.item_factors[i_idx] += lr * (err * u_vec - reg * i_vec)

    def fit(self, train_df: pd.DataFrame, all_items: set[int]) -> "BiasMFImplicitRanker":
        rng = np.random.default_rng(seed=self.random_state)

        user_ids = sorted(train_df["user_id"].astype(int).unique().tolist())
        item_ids = sorted(int(x) for x in all_items)

        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

        n_users = len(user_ids)
        n_items = len(item_ids)

        self.user_bias = np.zeros(n_users, dtype=float)
        self.item_bias = np.zeros(n_items, dtype=float)
        self.user_factors = rng.normal(0.0, 0.05, size=(n_users, self.n_factors))
        self.item_factors = rng.normal(0.0, 0.05, size=(n_items, self.n_factors))

        user_pos_items: dict[int, list[int]] = {}
        user_pos_sets: dict[int, set[int]] = {}

        grouped = train_df.groupby("user_id")["item_id"]
        for user_id, item_series in grouped:
            u_id = int(user_id)
            u_idx = self.user_id_to_idx.get(u_id)
            if u_idx is None:
                continue

            pos_list = []
            for item_id in item_series.astype(int).tolist():
                i_idx = self.item_id_to_idx.get(item_id)
                if i_idx is not None:
                    pos_list.append(i_idx)

            if not pos_list:
                continue

            user_pos_items[u_idx] = pos_list
            user_pos_sets[u_idx] = set(pos_list)

        all_item_indices = np.arange(n_items, dtype=np.int64)
        user_indices = np.array(sorted(user_pos_items.keys()), dtype=np.int64)

        for epoch in range(self.epochs):
            rng.shuffle(user_indices)
            updates = 0

            for u_idx in user_indices:
                pos_list = user_pos_items[int(u_idx)]
                pos_set = user_pos_sets[int(u_idx)]

                for pos_i_idx in pos_list:
                    self._sgd_update(int(u_idx), int(pos_i_idx), 1.0)
                    updates += 1

                    neg_drawn = 0
                    while neg_drawn < self.negatives_per_positive:
                        neg_i_idx = int(rng.choice(all_item_indices))
                        if neg_i_idx in pos_set:
                            continue
                        self._sgd_update(int(u_idx), neg_i_idx, 0.0)
                        updates += 1
                        neg_drawn += 1

            print(f"[BiasMF] epoch {epoch + 1}/{self.epochs} finished, updates={updates}")

        return self

    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.user_bias is not None
        assert self.item_bias is not None
        assert self.user_factors is not None
        assert self.item_factors is not None

        scores = np.full(shape=item_ids.shape, fill_value=self.global_bias, dtype=float)

        u_idx = self.user_id_to_idx.get(int(user_id))
        if u_idx is not None:
            u_bias = self.user_bias[u_idx]
            u_vec = self.user_factors[u_idx]
            scores += u_bias
        else:
            u_vec = np.zeros(self.n_factors, dtype=float)

        i_indices = np.array([self.item_id_to_idx.get(int(iid), -1) for iid in item_ids], dtype=int)
        known_mask = i_indices >= 0
        if known_mask.any():
            known_i = i_indices[known_mask]
            scores[known_mask] += self.item_bias[known_i]
            scores[known_mask] += self.item_factors[known_i] @ u_vec

        return scores

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "BiasMFImplicitRanker":
        return joblib.load(path)
