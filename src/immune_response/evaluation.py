"""Evaluation for unseen drug--cell-type combinations.

Rows, not individual cells, are the experimental units here. Gene rankings are
descriptive effect-size checks; they do not establish significant differential
expression, pathway recovery, donor generalization, or causal mechanism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_meta(meta: pd.DataFrame) -> None:
    if not {"drug", "cell_type"}.issubset(meta.columns):
        raise ValueError("Metadata must contain drug and cell_type columns.")
    if meta.empty or meta[["drug", "cell_type"]].isna().any().any():
        raise ValueError("Metadata must be nonempty with no missing labels.")
    if meta.duplicated(["drug", "cell_type"]).any():
        raise ValueError("Each drug--cell-type combination must have one row.")


def make_splits(meta: pd.DataFrame, n_splits: int = 3, seed: int = 42):
    """Return positional train/test indices for complete combination holdouts.

    Every row is tested exactly once. Both category labels in every test row
    remain represented in training. This is a missing-combination benchmark,
    not a test of entirely unseen compounds or entirely unseen cell types.
    """
    _validate_meta(meta)
    if not isinstance(n_splits, int) or not 2 <= n_splits <= len(meta):
        raise ValueError("n_splits must be an integer between 2 and the row count.")
    for column in ("drug", "cell_type"):
        if meta.groupby(column, observed=True).size().min() < 2:
            raise ValueError(f"Each {column} needs at least two combinations.")

    rng = np.random.default_rng(seed)
    groups = list(meta.groupby("drug", sort=True, observed=True).indices.values())
    # Round-robin within each drug prevents an entire compound entering one
    # test fold. Retry the randomized offsets to protect rare cell-type labels.
    for _ in range(512):
        assignment = np.empty(len(meta), dtype=int)
        for group in groups:
            rows = rng.permutation(group)
            assignment[rows] = (np.arange(len(rows)) + rng.integers(n_splits)) % n_splits
        if len(np.unique(assignment)) != n_splits:
            continue
        if any(
            len(np.unique(assignment[rows])) < 2
            for rows in meta.groupby("cell_type", observed=True).indices.values()
        ):
            continue
        splits = []
        for fold in range(n_splits):
            train = np.flatnonzero(assignment != fold)
            test = np.flatnonzero(assignment == fold)
            train_pairs = set(map(tuple, meta.iloc[train][["drug", "cell_type"]].to_numpy()))
            test_pairs = set(map(tuple, meta.iloc[test][["drug", "cell_type"]].to_numpy()))
            if train_pairs & test_pairs:
                raise AssertionError("Train/test combination leakage.")
            for column in ("drug", "cell_type"):
                if not set(meta.iloc[test][column]).issubset(set(meta.iloc[train][column])):
                    raise AssertionError(f"Unseen test {column} category.")
            splits.append((train, test))
        tested = np.concatenate([test for _, test in splits])
        if not np.array_equal(np.sort(tested), np.arange(len(meta))):
            raise AssertionError("Each row must be tested exactly once.")
        return splits
    raise ValueError("Could not construct category-preserving folds; reduce n_splits.")


def score_predictions(meta, y_true, y_pred, model, fold, gene_names=None):
    """Compute per-combination metrics with explicit undefined-score handling.

    Top-50 metrics use min(50, number of nonzero true-effect genes). Pearson is
    NaN for a constant true/predicted vector. Top-gene overlap is NaN if the
    predicted absolute effects are all equal (there is no informative ranking).
    Sign accuracy is zero for zero predictions on nonzero true-effect genes.
    Ties in informative rankings use the supplied gene order deterministically.
    """
    _validate_meta(meta)
    truth, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.ndim != 2 or truth.shape != pred.shape or truth.shape[0] != len(meta):
        raise ValueError("Targets/predictions must have matching (rows, genes) shapes.")
    if truth.shape[1] == 0 or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Targets/predictions must contain finite values and at least one gene.")
    if gene_names is not None:
        if len(gene_names) != truth.shape[1] or len(set(gene_names)) != len(gene_names):
            raise ValueError("gene_names must be unique and match the target width.")

    centered_true = truth - truth.mean(axis=1, keepdims=True)
    centered_pred = pred - pred.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(centered_true, axis=1) * np.linalg.norm(centered_pred, axis=1)
    pearson = np.divide(
        np.sum(centered_true * centered_pred, axis=1), denom,
        out=np.full(len(meta), np.nan), where=denom > 1e-12,
    )
    rows = meta[["drug", "cell_type"]].reset_index(drop=True).copy()
    rows["model"], rows["fold"] = model, fold
    rows["rmse"] = np.sqrt(np.mean((truth - pred) ** 2, axis=1))
    rows["pearson"] = np.clip(pearson, -1, 1)
    overlaps, signs = [], []
    for observed, predicted in zip(truth, pred):
        order = np.argsort(-np.abs(observed), kind="stable")
        top = order[np.abs(observed[order]) > 1e-12][:50]
        if not len(top):
            overlaps.append(np.nan)
            signs.append(np.nan)
            continue
        signs.append(float(np.mean(np.sign(observed[top]) == np.sign(predicted[top]))))
        if np.ptp(np.abs(predicted)) <= 1e-12:
            overlaps.append(np.nan)
        else:
            selected = np.argsort(-np.abs(predicted), kind="stable")[:len(top)]
            overlaps.append(len(np.intersect1d(top, selected)) / len(top))
    rows["top50_overlap"] = overlaps
    rows["top50_sign_accuracy"] = signs
    return rows


def bootstrap_rmse_improvement(
    scores: pd.DataFrame, baseline: str, challenger: str,
    n_boot: int = 2000, seed: int = 42, confidence: float = 0.95,
):
    """Paired drug-cluster bootstrap of mean row-RMSE improvement.

    Positive values favor the challenger. Resampling an entire compound keeps
    its cell-type rows together. This quantifies variability over sampled drugs,
    conditional on these donors, folds and fitted models; it is not a patient-
    level confidence interval or evidence of biological replicate variation.
    """
    if n_boot < 2 or not 0 < confidence < 1:
        raise ValueError("Require n_boot >= 2 and confidence strictly between 0 and 1.")
    required = {"drug", "cell_type", "fold", "model", "rmse"}
    if not required.issubset(scores.columns):
        raise ValueError(f"Scores require columns {sorted(required)}.")
    key = ["drug", "cell_type", "fold"]
    a = scores.loc[scores.model == baseline, key + ["rmse"]]
    b = scores.loc[scores.model == challenger, key + ["rmse"]]
    if a.empty or b.empty or a.duplicated(key).any() or b.duplicated(key).any():
        raise ValueError("Both models need one score per identical evaluation row.")
    joined = a.merge(b, on=key, how="outer", suffixes=("_baseline", "_challenger"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Models must have identical evaluation rows; no implicit dropping.")
    if not np.isfinite(joined[["rmse_baseline", "rmse_challenger"]]).all().all():
        raise ValueError("RMSE scores must be finite.")
    joined["improvement"] = joined.rmse_baseline - joined.rmse_challenger
    groups = joined.groupby("drug", sort=True).improvement.agg(["sum", "count"])
    if len(groups) < 2:
        raise ValueError("At least two drug clusters are required.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups), size=(n_boot, len(groups)))
    improvements = groups["sum"].to_numpy()[draws].sum(axis=1) / groups["count"].to_numpy()[draws].sum(axis=1)
    low, high = np.quantile(improvements, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {
        "baseline": baseline, "challenger": challenger,
        "mean_rmse_improvement": float(joined.improvement.mean()),
        "ci_low": float(low), "ci_high": float(high),
        "confidence": confidence, "n_drugs": len(groups),
        "n_rows": len(joined), "n_boot": n_boot, "seed": seed,
        "estimand": "mean per-combination RMSE improvement, drug-cluster bootstrap",
    }
