"""Run development selection, frozen final evaluation, and secondary acquisition."""

from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
from immune_response.data import download, load_profiles, assert_transfer, write_manifest
from immune_response.models import ZeroBaseline, DrugMeanBaseline, AdditiveRidge, PCARidge, TorchInteractionModel
from immune_response.evaluation import score_predictions, bootstrap_rmse_improvement

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
PLAN = json.loads((ROOT / "study_plan.json").read_text())
SEEDS = PLAN["torch_fixed"]["seeds"]


def torch_predict(train, y, target, config):
    predictions = []
    for seed in SEEDS:
        model = TorchInteractionModel(seed=seed, **config,
                                      **{k: v for k, v in PLAN["torch_fixed"].items() if k != "seeds"})
        predictions.append(model.fit(train, y).predict(target))
    return np.mean(predictions, axis=0), np.std(predictions, axis=0, ddof=1)


def develop(meta, y, genes):
    train = np.flatnonzero(meta.split.isin(["train", "control"]))
    valid = np.flatnonzero(meta.split == "public_test")
    a, b = meta.iloc[train], meta.iloc[valid]
    assert_transfer(a, b)
    scores, configs = [], {}
    baseline_models = {"No change": ZeroBaseline(), "Drug mean": DrugMeanBaseline()}
    for label, model in baseline_models.items():
        pred = model.fit(a, y[train]).predict(b)
        scores.append(score_predictions(b, y[valid], pred, label, "validation", genes))
    for alpha in PLAN["ridge_alphas"]:
        label = f"Ridge alpha={alpha}"
        pred = AdditiveRidge(alpha=alpha).fit(a, y[train]).predict(b)
        scores.append(score_predictions(b, y[valid], pred, label, "validation", genes))
        configs[label] = {"alpha": alpha}
        label = f"PCA ridge alpha={alpha}"
        pred = PCARidge(alpha=alpha, n_components=24).fit(a, y[train]).predict(b)
        scores.append(score_predictions(b, y[valid], pred, label, "validation", genes))
        configs[label] = {"alpha": alpha, "n_components": 24}
    for config in PLAN["torch_candidates"]:
        label = f"PyTorch epochs={config['epochs']} decay={config['weight_decay']}"
        pred, _ = torch_predict(a, y[train], b, config)
        scores.append(score_predictions(b, y[valid], pred, label, "validation", genes))
        configs[label] = config
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(OUT / "validation_scores.csv", index=False)
    means = scores.groupby("model").rmse.mean()
    ridge = means[means.index.str.startswith("Ridge")].idxmin()
    pca_ridge = means[means.index.str.startswith("PCA ridge")].idxmin()
    neural = means[means.index.str.startswith("PyTorch")].idxmin()
    selection = {"ridge": configs[ridge], "pca_ridge": configs[pca_ridge], "torch": configs[neural],
                 "criterion": "mean validation profile RMSE",
                 "plan_sha256": hashlib.sha256((ROOT / "study_plan.json").read_bytes()).hexdigest(),
                 "n_training": len(train), "n_validation": len(valid),
                 "validation_means": means.to_dict()}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("Selected on validation:", ridge, pca_ridge, neural, flush=True)


def final(meta, y, genes):
    selection = json.loads((OUT / "selection.json").read_text())
    if selection["plan_sha256"] != hashlib.sha256((ROOT / "study_plan.json").read_bytes()).hexdigest():
        raise ValueError("Study plan changed after selection")
    test, truth, test_genes = load_profiles(ROOT / "data/de_test.h5ad")
    if not np.array_equal(genes, test_genes):
        raise ValueError("Gene order mismatch")
    assert_transfer(meta, test)
    scores, predictions = [], {}
    models = {"No change": ZeroBaseline(), "Drug mean": DrugMeanBaseline(),
              "Additive ridge": AdditiveRidge(**selection["ridge"]),
              "PCA ridge": PCARidge(**selection["pca_ridge"])}
    for label, model in models.items():
        predictions[label] = model.fit(meta, y).predict(test)
    predictions["PyTorch ensemble"], spread = torch_predict(meta, y, test, selection["torch"])
    for label, pred in predictions.items():
        scores.append(score_predictions(test, truth, pred, label, "private_test", genes))
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(OUT / "test_scores.csv", index=False)
    metrics = ["rmse", "pearson", "top50_overlap", "top50_sign_accuracy"]
    summary = scores.groupby("model", sort=False)[metrics].mean()
    summary.to_csv(OUT / "summary.csv")
    intervals = [bootstrap_rmse_improvement(scores, baseline, challenger)
                 for baseline, challenger in [("No change", "Drug mean"),
                     ("Drug mean", "Additive ridge"), ("Drug mean", "PyTorch ensemble"),
                     ("Additive ridge", "PyTorch ensemble"), ("PCA ridge", "PyTorch ensemble")]]
    (OUT / "paired_intervals.json").write_text(json.dumps(intervals, indent=2) + "\n")
    # Detailed predictions and truth stay local; text metrics/figures are shareable.
    np.savez_compressed(ROOT / "data/test_predictions.npz", truth=truth, genes=genes,
                        uncertainty=spread, **{k.replace(" ", "_"): v for k, v in predictions.items()})
    test.to_csv(OUT / "test_conditions.csv", index=False)
    uncertainty = test[["drug", "cell_type"]].copy()
    uncertainty["ensemble_disagreement"] = np.sqrt(np.mean(spread ** 2, axis=1))
    uncertainty["rmse"] = np.sqrt(np.mean((truth - predictions["PyTorch ensemble"]) ** 2, axis=1))
    uncertainty.to_csv(OUT / "uncertainty_diagnostic.csv", index=False)
    print(summary.round(4).to_string(), flush=True)


def acquisition(meta, y):
    from immune_response.active_learning import run_active_learning
    results = run_active_learning(meta, y, seeds=range(5), n_rounds=4, batch_size=30,
                                  evaluation_indices=np.flatnonzero(meta.split == "public_test"))
    results.to_csv(OUT / "active_learning.csv", index=False)
    provenance = {key: value for key, value in results.attrs.items() if key != "acquisition_history"}
    provenance["acquisition_history"] = results.attrs.get("acquisition_history", [])
    (OUT / "active_learning_protocol.json").write_text(json.dumps(provenance, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x), indent=2) + "\n")
    print("Offline acquisition simulation complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "develop", "final", "acquisition"], default="all")
    args = parser.parse_args()
    start = time.time()
    OUT.mkdir(exist_ok=True)
    download(ROOT / "data")
    meta, y, genes = load_profiles(ROOT / "data/de_train.h5ad")
    if args.stage in ("all", "develop"):
        develop(meta, y, genes)
    if args.stage in ("all", "final"):
        final(meta, y, genes)
    if args.stage in ("all", "acquisition"):
        acquisition(meta, y)
    write_manifest(ROOT / "data", OUT / "data_manifest.json")
    import sklearn, torch, scipy, h5py, seaborn, matplotlib
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__, "torch": torch.__version__, "scipy": scipy.__version__,
                "h5py": h5py.__version__, "seaborn": seaborn.__version__, "matplotlib": matplotlib.__version__}
    (OUT / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    print(f"Completed stage {args.stage} in {time.time() - start:.1f}s", flush=True)
