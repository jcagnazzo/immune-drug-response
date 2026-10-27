"""Small, auditable models for held-out drug–cell-type response profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


def _metadata(meta: pd.DataFrame) -> pd.DataFrame:
    columns = ["drug", "cell_type"]
    if not set(columns).issubset(meta.columns):
        raise ValueError("Metadata must contain drug and cell_type columns")
    if meta[columns].isna().any().any():
        raise ValueError("Drug and cell_type labels cannot be missing")
    return meta[columns].astype(str)


def _training_data(meta: pd.DataFrame, y: np.ndarray):
    meta = _metadata(meta)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2 or y.shape[0] != len(meta) or min(y.shape) == 0:
        raise ValueError("Responses must be a nonempty rows-by-genes matrix")
    if not np.isfinite(y).all():
        raise ValueError("Responses must be finite")
    return meta, y


class ZeroBaseline:
    """Predict no treatment-associated change in any gene."""

    def fit(self, meta: pd.DataFrame, y: np.ndarray):
        _, y = _training_data(meta, y)
        self.n_outputs_ = y.shape[1]
        return self

    def predict(self, meta: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(_metadata(meta)), self.n_outputs_), dtype=np.float32)


class DrugMeanBaseline:
    """Transfer a drug's mean measured response across training cell types.

    Entirely unseen drugs fall back to the training global mean; that fallback
    is not evidence of useful unseen-drug generalization.
    """

    def fit(self, meta: pd.DataFrame, y: np.ndarray):
        meta, y = _training_data(meta, y)
        self.global_mean_ = y.mean(axis=0)
        drugs = meta["drug"].to_numpy()
        self.drug_means_ = {drug: y[drugs == drug].mean(axis=0) for drug in np.unique(drugs)}
        return self

    def predict(self, meta: pd.DataFrame) -> np.ndarray:
        drugs = _metadata(meta)["drug"]
        if len(drugs) == 0:
            return np.empty((0, len(self.global_mean_)), dtype=np.float32)
        return np.stack([self.drug_means_.get(drug, self.global_mean_) for drug in drugs])


class AdditiveRidge:
    """Regularized additive effects of drug identity and cell-type identity."""

    def __init__(self, alpha: float = 10.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def fit(self, meta: pd.DataFrame, y: np.ndarray):
        meta, y = _training_data(meta, y)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        x = self.encoder_.fit_transform(meta)
        self.regressor_ = Ridge(alpha=self.alpha, solver="cholesky").fit(x, y)
        return self

    def predict(self, meta: pd.DataFrame) -> np.ndarray:
        return self.regressor_.predict(self.encoder_.transform(_metadata(meta)))


class PCARidge(AdditiveRidge):
    """Linear comparator with the neural model's training-only target PCA.

    This controls for response compression, while fitting coefficients by
    regularized least squares rather than neural optimization.
    """

    def __init__(self, alpha: float = 10.0, n_components: int = 24, seed: int = 42):
        super().__init__(alpha)
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components, self.seed = n_components, seed

    def fit(self, meta: pd.DataFrame, y: np.ndarray):
        meta, y = _training_data(meta, y)
        if len(meta) < 2:
            raise ValueError("At least two training profiles are needed for PCA")
        self.pca_ = PCA(n_components=min(self.n_components, len(y) - 1, y.shape[1]),
                        svd_solver="randomized", random_state=self.seed)
        return super().fit(meta, self.pca_.fit_transform(y).astype(np.float32))

    def predict(self, meta: pd.DataFrame) -> np.ndarray:
        return self.pca_.inverse_transform(super().predict(meta)).astype(np.float32)


class TorchInteractionModel:
    """Regularized nonlinear response model with training-only target PCA.

    A linear skip connection and one hidden layer map drug/cell-type one-hot
    features to response principal components. PCA, category vocabulary, and
    latent scaling are fitted only on rows passed to ``fit``. Training uses a
    fixed epoch budget, never the outer test labels. This is a deliberately
    compact comparator, not a pretrained biological foundation model.
    """

    def __init__(self, seed: int = 42, epochs: int = 250, hidden: int = 64,
                 n_components: int = 24, learning_rate: float = 0.01,
                 weight_decay: float = 0.05):
        if min(epochs, hidden, n_components) < 1 or learning_rate <= 0 or weight_decay < 0:
            raise ValueError("Model sizes, epochs and learning rate must be positive")
        self.seed, self.epochs, self.hidden = seed, epochs, hidden
        self.n_components = n_components
        self.learning_rate, self.weight_decay = learning_rate, weight_decay

    def fit(self, meta: pd.DataFrame, y: np.ndarray):
        import torch
        from torch import nn

        meta, y = _training_data(meta, y)
        if len(meta) < 2:
            raise ValueError("At least two training profiles are needed for PCA")
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        x = self.encoder_.fit_transform(meta)
        self.pca_ = PCA(n_components=min(self.n_components, len(y) - 1, y.shape[1]),
                        svd_solver="randomized", random_state=self.seed)
        z = self.pca_.fit_transform(y).astype(np.float32)
        self.latent_scale_ = max(float(np.sqrt(np.mean(z ** 2))), 1e-6)
        z /= self.latent_scale_

        class ResponseNetwork(nn.Module):
            def __init__(self, inputs, hidden, outputs):
                super().__init__()
                self.linear = nn.Linear(inputs, outputs)
                self.interaction = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(),
                                                 nn.Linear(hidden, outputs))

            def forward(self, inputs):
                return self.linear(inputs) + self.interaction(inputs)

        # Restrict threads for predictable runtimes on these small matrices.
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network_ = ResponseNetwork(x.shape[1], self.hidden, z.shape[1])
            optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(z)
            self.training_loss_ = []
            self.network_.train()
            for _ in range(self.epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(self.network_(inputs), targets)
                loss.backward()
                optimizer.step()
                self.training_loss_.append(float(loss.detach()))
        self.network_.eval()
        return self

    def predict(self, meta: pd.DataFrame) -> np.ndarray:
        import torch

        x = self.encoder_.transform(_metadata(meta)).astype(np.float32)
        with torch.no_grad():
            z = self.network_(torch.from_numpy(x)).numpy() * self.latent_scale_
        return self.pca_.inverse_transform(z).astype(np.float32)


def ensemble_predictions(models, meta: pd.DataFrame):
    """Return ensemble mean and per-gene disagreement, not calibrated uncertainty."""
    if len(models) < 2:
        raise ValueError("At least two fitted models are required")
    predictions = np.stack([model.predict(meta) for model in models])
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=1)
