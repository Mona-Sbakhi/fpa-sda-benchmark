"""Feature pipelines. All transforms are fitted on the optimisation (training) subset only."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAFeatures:
    """Flatten -> scale to [0,1] -> PCA(k) -> optional standardisation.

    `standardize=False` reproduces the original notebook (raw PCA scores, which saturate the
    softmax when weights are drawn from [-1, 1]); `standardize=True` is the revised default.
    """

    def __init__(self, n_components: int = 1000, standardize: bool = True, seed: int = 42):
        self.pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
        self.scaler = StandardScaler() if standardize else None

    @staticmethod
    def _flat(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1).astype(np.float32) / 255.0

    def fit(self, X: np.ndarray) -> "PCAFeatures":
        Z = self.pca.fit_transform(self._flat(X))
        if self.scaler is not None:
            self.scaler.fit(Z)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.pca.transform(self._flat(X))
        if self.scaler is not None:
            Z = self.scaler.transform(Z)
        return Z.astype(np.float32)

    @property
    def explained_variance(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


def load_precomputed(path: str) -> dict:
    """Load CNN features produced by scripts/extract_cnn_features.py (npz with
    F_train, y_train, F_test, y_test, train_files, test_files)."""
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}
