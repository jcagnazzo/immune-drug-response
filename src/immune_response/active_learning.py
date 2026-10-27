"""Offline acquisition experiment using label-free, design-based ridge leverage.

The evaluation set never enters the acquisition pool. Pool responses are only
indexed after selecting rows; scores depend on design features, not responses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from .models import _training_data


def _category_cover(meta: pd.DataFrame, indices: np.ndarray, rng) -> np.ndarray:
    """Pick rows covering every drug and cell type without looking at outcomes."""
    remaining = {column: set(meta.iloc[indices][column]) for column in ("drug", "cell_type")}
    chosen = []
    for index in rng.permutation(indices):
        row = meta.iloc[index]
        if any(row[column] in remaining[column] for column in remaining):
            chosen.append(index)
            for column in remaining:
                remaining[column].discard(row[column])
    return np.asarray(chosen, dtype=int)


def ridge_leverage(x_labeled: np.ndarray, x_pool: np.ndarray, alpha: float = 10.0):
    """Unit-noise predictive-variance proxy for a ridge design, not calibration."""
    covariance = x_labeled.T @ x_labeled + alpha * np.eye(x_labeled.shape[1])
    solved = np.linalg.solve(covariance, x_pool.T)
    return np.einsum("ij,ji->i", x_pool, solved)


def run_active_learning(meta: pd.DataFrame, y: np.ndarray, seeds=range(5),
                        n_rounds: int = 4, batch_size: int = 30,
                        alpha: float = 10.0, evaluation_fraction: float = 0.2,
                        split_seed: int = 2026, evaluation_indices=None) -> pd.DataFrame:
    """Compare random and leverage acquisition at a fixed evaluation set.

    The caller must pass development data only, excluding the main study test
    profiles. Every replicate starts with all category labels represented.
    Both strategies share initial labels, budgets, and evaluation profiles.
    Supply ``evaluation_indices`` for a prespecified positional evaluation
    split, such as the official public-validation profiles. All its drug and
    cell-type labels must remain represented outside that evaluation set.
    The ridge model predicts every gene; no outcome-based feature selection
    or target PCA is used. Within each round, the whole batch is ranked from
    the current labeled design before any of its responses are revealed.
    """
    meta, y = _training_data(meta, y)
    meta = meta.reset_index(drop=True)
    if n_rounds < 0 or batch_size < 1 or alpha <= 0 or not 0 < evaluation_fraction < 1:
        raise ValueError("Invalid acquisition budget, regularization, or evaluation fraction")
    indices = np.arange(len(meta))
    rng_split = np.random.default_rng(split_seed)
    if evaluation_indices is None:
        protected = _category_cover(meta, indices, rng_split)
        available = np.setdiff1d(indices, protected)
        if len(available) < 2:
            raise ValueError("Insufficient profiles to separate evaluation and acquisition")
        n_evaluation = min(max(1, round(len(meta) * evaluation_fraction)), len(available) - 1)
        evaluation = np.sort(rng_split.choice(available, size=n_evaluation, replace=False))
    else:
        evaluation = np.asarray(evaluation_indices)
        if (evaluation.ndim != 1 or len(evaluation) == 0
                or not np.issubdtype(evaluation.dtype, np.integer)
                or len(np.unique(evaluation)) != len(evaluation)
                or np.any(evaluation < 0) or np.any(evaluation >= len(meta))):
            raise ValueError("evaluation_indices must be unique valid positional integers")
        evaluation = np.sort(evaluation)
    development = np.setdiff1d(indices, evaluation)
    if len(development) < 2:
        raise ValueError("At least two acquisition/development profiles are required")
    for column in ("drug", "cell_type"):
        if not set(meta.iloc[evaluation][column]) <= set(meta.iloc[development][column]):
            raise ValueError(f"Evaluation contains an unseen {column} category")
    encoder = OneHotEncoder(handle_unknown="error", sparse_output=False, dtype=np.float64)
    encoder.fit(meta.iloc[development])
    x = encoder.transform(meta)
    initial_size = min(meta.iloc[development].drug.nunique()
                       + meta.iloc[development].cell_type.nunique(), len(development) - 1)
    records, histories = [], {}

    for seed in seeds:
        rng_initial = np.random.default_rng(seed)
        initial = _category_cover(meta, development, rng_initial)
        # Category coverage determines a lower bound; all replicates share
        # the same nominal initial budget in the usual crossed design.
        if len(initial) < initial_size:
            extra = rng_initial.choice(np.setdiff1d(development, initial),
                                       initial_size - len(initial), replace=False)
            initial = np.concatenate([initial, extra])
        for strategy in ("random", "ridge_leverage"):
            rng = np.random.default_rng(seed)
            labeled = initial.copy()
            pool = np.setdiff1d(development, labeled)
            history = {"initial": initial.tolist(), "batches": []}
            histories[f"{seed}/{strategy}"] = history
            for round_index in range(n_rounds + 1):
                model = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky")
                model.fit(x[labeled], y[labeled])
                prediction = model.predict(x[evaluation])
                rmse = float(np.sqrt(np.mean((prediction - y[evaluation]) ** 2, axis=1)).mean())
                records.append({"seed": int(seed), "strategy": strategy,
                                "n_labeled": len(labeled), "rmse": rmse,
                                "n_evaluation": len(evaluation)})
                if round_index == n_rounds or len(pool) == 0:
                    break
                size = min(batch_size, len(pool))
                if strategy == "random":
                    selected = rng.choice(pool, size=size, replace=False)
                else:
                    scores = ridge_leverage(x[labeled], x[pool], alpha)
                    # Random tie-breaking uses features only; pool labels
                    # are not used to choose which experiments to reveal.
                    order = np.lexsort((rng.random(len(pool)), -scores))
                    selected = pool[order[:size]]
                labeled = np.concatenate([labeled, selected])
                history["batches"].append(selected.tolist())
                pool = np.setdiff1d(pool, selected)
    result = pd.DataFrame(records)
    result.attrs["evaluation_indices"] = evaluation.tolist()
    result.attrs["development_indices"] = development.tolist()
    result.attrs["split_seed"] = split_seed
    result.attrs["variance_interpretation"] = "uncalibrated design-based ridge leverage"
    result.attrs["acquisition_history"] = histories
    return result
