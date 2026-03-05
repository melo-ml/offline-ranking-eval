from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class LogisticRegressionRanker:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=300)
        self.user_counts: dict[int, float] = {}
        self.item_counts: dict[int, float] = {}
        self.item_genres: dict[int, np.ndarray] = {}
        self.user_genre_prefs: dict[int, np.ndarray] = {}
        self.genre_to_idx: dict[str, int] = {}

    @staticmethod
    def _load_item_genres(movies_path: Path) -> tuple[dict[int, np.ndarray], dict[str, int]]:
        if not movies_path.exists():
            raise FileNotFoundError(f"Missing movies metadata file: {movies_path}")

        movies_df = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1",
        )

        genres_list = []
        for raw_genres in movies_df["genres"].fillna(""):
            tokens = [g for g in str(raw_genres).split("|") if g and g != "(no genres listed)"]
            genres_list.extend(tokens)

        unique_genres = sorted(set(genres_list))
        genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}

        item_genres: dict[int, np.ndarray] = {}
        dim = len(genre_to_idx)
        for row in movies_df.itertuples(index=False):
            item_id = int(row.item_id)
            vec = np.zeros(dim, dtype=float)
            tokens = [g for g in str(row.genres).split("|") if g and g != "(no genres listed)"]
            for token in tokens:
                idx = genre_to_idx.get(token)
                if idx is not None:
                    vec[idx] = 1.0
            item_genres[item_id] = vec

        return item_genres, genre_to_idx

    def fit(
        self,
        train_df: pd.DataFrame,
        all_items: set[int],
        movies_path: Path,
        negatives_per_positive: int = 3,
        random_state: int = 42,
    ) -> "LogisticRegressionRanker":
        rng = np.random.default_rng(seed=random_state)

        user_counts_series = train_df["user_id"].value_counts()
        item_counts_series = train_df["item_id"].value_counts()

        self.user_counts = {int(uid): float(cnt) for uid, cnt in user_counts_series.items()}
        self.item_counts = {int(iid): float(cnt) for iid, cnt in item_counts_series.items()}
        self.item_genres, self.genre_to_idx = self._load_item_genres(movies_path)

        genre_dim = len(self.genre_to_idx)
        zero_genre_vec = np.zeros(genre_dim, dtype=float)

        all_items_arr = np.array(sorted(all_items), dtype=np.int64)
        user_to_items = train_df.groupby("user_id")["item_id"].apply(list)

        # User profile is mean genre vector over historical positive interactions.
        for user_id, positives in user_to_items.items():
            user_id_int = int(user_id)
            vectors = [self.item_genres.get(int(item_id), zero_genre_vec) for item_id in positives]
            if vectors:
                self.user_genre_prefs[user_id_int] = np.mean(np.vstack(vectors), axis=0)
            else:
                self.user_genre_prefs[user_id_int] = zero_genre_vec

        features = []
        labels = []

        for user_id, positives in user_to_items.items():
            user_id_int = int(user_id)
            user_cnt = self.user_counts.get(user_id_int, 0.0)
            user_pref = self.user_genre_prefs.get(user_id_int, zero_genre_vec)

            seen_items = np.array(sorted(set(int(x) for x in positives)), dtype=np.int64)
            neg_candidates = np.setdiff1d(all_items_arr, seen_items, assume_unique=False)
            if neg_candidates.size == 0:
                continue

            for pos_item in positives:
                pos_item_int = int(pos_item)
                pos_item_vec = self.item_genres.get(pos_item_int, zero_genre_vec)
                pos_genre_match = float(np.dot(user_pref, pos_item_vec))
                features.append([user_cnt, self.item_counts.get(pos_item_int, 0.0), pos_genre_match])
                labels.append(1)

                sample_size = min(negatives_per_positive, int(neg_candidates.size))
                neg_items = rng.choice(neg_candidates, size=sample_size, replace=False)
                for neg_item in np.atleast_1d(neg_items):
                    neg_item_int = int(neg_item)
                    neg_item_vec = self.item_genres.get(neg_item_int, zero_genre_vec)
                    neg_genre_match = float(np.dot(user_pref, neg_item_vec))
                    features.append([user_cnt, self.item_counts.get(neg_item_int, 0.0), neg_genre_match])
                    labels.append(0)

        if not features:
            raise ValueError("No training examples generated for LogisticRegressionRanker.")

        X = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)

        self.model.fit(X, y)
        return self

    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        user_cnt = self.user_counts.get(int(user_id), 0.0)
        genre_dim = len(self.genre_to_idx)
        zero_genre_vec = np.zeros(genre_dim, dtype=float)
        user_pref = self.user_genre_prefs.get(int(user_id), zero_genre_vec)
        item_cnts = np.array([self.item_counts.get(int(iid), 0.0) for iid in item_ids], dtype=float)
        genre_match = np.array(
            [float(np.dot(user_pref, self.item_genres.get(int(iid), zero_genre_vec))) for iid in item_ids],
            dtype=float,
        )
        user_cnts = np.full(shape=item_cnts.shape, fill_value=user_cnt, dtype=float)
        X = np.column_stack([user_cnts, item_cnts, genre_match])

        if X.size == 0:
            return np.array([], dtype=float)

        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "LogisticRegressionRanker":
        return joblib.load(path)
