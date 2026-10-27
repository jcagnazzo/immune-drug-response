"""Regenerate seaborn figures, Reactome summaries and a self-contained report."""

from pathlib import Path
import base64
import html
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from immune_response.data import load_profiles
from immune_response.pathways import download_reactome, read_reactome, pathway_responses, evaluate_pathways

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "reports", ROOT / "reports/figures"
MODELS = ["No change", "Drug mean", "Additive ridge", "PCA ridge", "PyTorch ensemble"]
COLORS = ["#b6bdc5", "#c79463", "#5b80a3", "#347e7b", "#153951"]
sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans",
              rc={"axes.spines.top": False, "axes.spines.right": False,
                  "axes.labelcolor": "#25384a", "text.color": "#153951", "grid.alpha": .25,
                  "figure.facecolor": "white", "axes.titleweight": "bold"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cluster_interval(frame, metric):
    grouped = frame.groupby("drug")[metric].agg(["sum", "count"])
    rng = np.random.default_rng(42)
    draws = rng.integers(len(grouped), size=(2000, len(grouped)))
    values = grouped["sum"].to_numpy()[draws].sum(1) / grouped["count"].to_numpy()[draws].sum(1)
    return np.quantile(values, [.025, .975])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(OUT / "test_scores.csv")
    summary = pd.read_csv(OUT / "summary.csv").set_index("model").loc[MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    for j, (metric, title, xlabel) in enumerate([
        ("rmse", "Prediction error", "Mean profile RMSE · lower is better"),
        ("top50_overlap", "Recovery of large gene responses", "Top-50 effect-gene overlap · higher is better")]):
        for i, model in enumerate(MODELS):
            value = summary.loc[model, metric]
            if not np.isfinite(value):
                continue
            lo, hi = cluster_interval(scores[scores.model == model], metric)
            axes[j].errorbar(value, i, xerr=[[max(0, value-lo)], [max(0, hi-value)]],
                             fmt="o", color=COLORS[i], capsize=3, markersize=8, linewidth=2)
        axes[j].set_yticks(range(5), MODELS if j == 0 else [])
        axes[j].invert_yaxis()
        axes[j].set_xlabel(xlabel)
        axes[j].set_title(title, loc="left", pad=17)
        if j == 1:
            axes[j].set_xlim(0, .4)
    fig.suptitle("Drug-response transfer across immune cell types", fontsize=17, x=.06, ha="left", y=1.16)
    fig.text(.06, 1.075, "151 held-out conditions · 76 compounds · B / myeloid cells · 5,317 genes", fontsize=10)
    fig.text(.06, -.07, "Intervals: 95% drug-cluster bootstrap. No-change gene ranking is undefined. Benchmark-provided processed outcomes.", fontsize=9)
    fig.tight_layout()
    save(fig, "01_model_comparison")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    sns.boxplot(data=scores, x="cell_type", y="rmse", hue="model", hue_order=MODELS,
                palette=COLORS, showfliers=False, ax=ax, linewidth=1)
    ax.set(title="Performance varies across biological contexts", xlabel="Evaluation cell type",
           ylabel="Per-condition RMSE")
    ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, "02_cell_types")

    uncertainty = pd.read_csv(OUT / "uncertainty_diagnostic.csv")
    rho = float(spearmanr(uncertainty.ensemble_disagreement, uncertainty.rmse).statistic)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.scatterplot(data=uncertainty, x="ensemble_disagreement", y="rmse", hue="cell_type",
                    palette=["#153951", "#36a399"], s=38, alpha=.7, ax=ax)
    ax.set(title=f"Does ensemble disagreement track error?  Spearman ρ = {rho:.2f}",
           xlabel="RMS disagreement across three neural models", ylabel="Per-condition prediction RMSE")
    ax.legend(title="Cell type")
    fig.text(.12, -.015, "Descriptive test-set diagnostic; disagreement is not a calibrated confidence interval.", fontsize=9)
    fig.tight_layout()
    save(fig, "03_uncertainty")

    acquisition = pd.read_csv(OUT / "active_learning.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    al_summary = acquisition.groupby(["strategy", "n_labeled"]).rmse.agg(["mean", "min", "max"]).reset_index()
    for strategy, label, color in [("random", "Random selection", "#8b969e"),
                                    ("ridge_leverage", "Design-based leverage", "#177b78")]:
        subset = al_summary[al_summary.strategy == strategy]
        sns.lineplot(data=subset, x="n_labeled", y="mean", label=label, color=color, marker="o", ax=ax)
        ax.fill_between(subset.n_labeled, subset["min"], subset["max"], color=color, alpha=.12)
    ax.set(title="Which experiment should be measured next?", xlabel="Labeled drug–cell-type conditions",
           ylabel="Mean profile RMSE")
    fig.text(.12, -.015, "Five initializations; shading is the seed range. Offline, label-free selection; final test is unused.", fontsize=9)
    fig.tight_layout()
    save(fig, "04_experiment_selection")
    al_summary.to_csv(OUT / "active_learning_summary.csv", index=False)

    bundle = np.load(ROOT / "data/test_predictions.npz")
    truth, genes = bundle["truth"], bundle["genes"]
    conditions = pd.read_csv(OUT / "test_conditions.csv")
    reactome = read_reactome(download_reactome(ROOT / "data"))
    true_path, pathway_info = pathway_responses(truth, genes, reactome)
    pathway_info.to_csv(OUT / "pathway_coverage.csv")
    path_scores, path_predictions = [], {}
    for model in MODELS:
        pred = bundle[model.replace(" ", "_")]
        evaluated = evaluate_pathways(truth, pred, genes, reactome)
        evaluated["model"] = model
        path_scores.append(pd.concat([conditions, evaluated], axis=1))
        path_predictions[model], _ = pathway_responses(pred, genes, reactome)
    pd.concat(path_scores).to_csv(OUT / "pathway_scores.csv", index=False)
    # Prespecified display rule: profile closest to median observed response norm.
    magnitude = np.sqrt(np.mean(truth ** 2, axis=1))
    example = int(np.argmin(np.abs(magnitude - np.median(magnitude))))
    top_path = true_path.iloc[example].abs().nlargest(10).index
    display = pd.DataFrame({"Observed": true_path.loc[example, top_path],
                           **{model: pred.loc[example, top_path] for model, pred in path_predictions.items()
                              if model in ["Drug mean", "PCA ridge", "PyTorch ensemble"]}})
    example_label = f"{conditions.iloc[example].drug} · {conditions.iloc[example].cell_type}"
    display.to_csv(OUT / "example_pathways.csv")
    gene_top = np.argsort(-np.abs(truth[example]))[:20]
    pd.DataFrame({"gene": genes[gene_top], "observed_logfc": truth[example, gene_top],
                  "pca_ridge_logfc": bundle["PCA_ridge"][example, gene_top],
                  "pytorch_logfc": bundle["PyTorch_ensemble"][example, gene_top]}).to_csv(OUT / "example_genes.csv", index=False)
    example_meta = {"row": example, "drug": conditions.iloc[example].drug,
                    "cell_type": conditions.iloc[example].cell_type,
                    "selection": "Closest to median observed profile response norm; top pathways chosen by observed absolute mean logFC"}
    (OUT / "example_selection.json").write_text(json.dumps(example_meta, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(10, 6.7))
    labels = ["\n".join(textwrap.wrap(pathway_info.loc[key, "name"], 37)) for key in top_path]
    sns.heatmap(display, cmap="vlag", center=0, annot=True, fmt=".2f", yticklabels=labels,
                cbar_kws={"label": "Mean log2 fold change of measured pathway genes"}, ax=ax)
    ax.set_title(f"Pathway response summaries: {example_label}", loc="left", pad=22)
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="x", rotation=20)
    fig.text(.02, -.015, "Descriptive means across overlapping gene sets; not enrichment, activation or causal validation. Median-effect example.", fontsize=8)
    fig.tight_layout()
    save(fig, "05_pathway_example")

    neural = summary.loc["PyTorch ensemble", "rmse"]
    reduction = 100 * (1 - neural / summary.loc["No change", "rmse"])
    ci = json.loads((OUT / "paired_intervals.json").read_text())[-1]
    table = summary.rename(columns={"rmse": "RMSE ↓", "pearson": "Pearson ↑", "top50_overlap": "Top-50 overlap ↑", "top50_sign_accuracy": "Top-50 sign accuracy ↑"})
    final_al = al_summary[al_summary.n_labeled == al_summary.n_labeled.max()].set_index("strategy")
    figures = []
    for name, caption in [
        ("01_model_comparison", "Figure 1. Aggregate accuracy and recovery of genes with large observed responses. Intervals resample whole compounds."),
        ("02_cell_types", "Figure 2. Error distributions by cell context. Boxes summarize conditions; whiskers omit plotted outliers."),
        ("03_uncertainty", "Figure 3. Ensemble disagreement versus actual error. This is an exploratory diagnostic, not uncertainty calibration."),
        ("04_experiment_selection", "Figure 4. Offline acquisition using public validation only. Shading shows the range over five initializations, not a confidence interval."),
        ("05_pathway_example", "Figure 5. A median-effect condition and its ten largest absolute pathway means. Overlapping pathways and incomplete gene coverage limit interpretation.")]:
        encoded = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
        figures.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>')
    sources = '<a href="https://github.com/openproblems-bio/task_perturbation_prediction">OP3 benchmark & processing</a> · <a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945">GSE279945</a> · <a href="https://reactome.org/download-data">Reactome</a>'
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Predicting drug responses across immune cell types</title><style>
    body{{margin:0;background:#f3f5f6;color:#223748;font:17px/1.65 system-ui,sans-serif}}main{{max-width:1040px;margin:auto;padding:55px 38px;background:white}}
    h1{{font-size:42px;line-height:1.13;letter-spacing:-1.4px;margin:14px 0 20px}}h2{{font-size:26px;margin-top:44px}}.eyebrow{{color:#177b78;font-size:12px;letter-spacing:2px;font-weight:700}}
    .lead{{font-size:21px;max-width:850px}}.callout{{border-left:4px solid #177b78;background:#edf5f3;padding:20px 24px;margin:28px 0}}
    table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce3e7;text-align:right}}th:first-child{{text-align:left}}
    figure{{margin:35px 0}}img{{max-width:100%;height:auto}}figcaption,.small{{font-size:13px;color:#5d6e79}}a{{color:#177b78}}code{{font-size:14px;background:#f1f3f5;padding:3px 6px}}li{{margin:8px 0}}
    @media(max-width:650px){{main{{padding:25px 18px}}h1{{font-size:31px}}table{{font-size:11px}}th,td{{padding:6px}}}}
    @media print{{main{{padding:0}}body{{font-size:11pt}}figure{{break-inside:avoid}}}}
    </style><main><div class="eyebrow">COMPUTATIONAL BIOLOGY · EXPLORATORY RESEARCH STUDY · OCTOBER 2026</div>
    <h1>Predicting drug responses<br>across immune cell types</h1>
    <p class="lead">Can responses measured in one immune-cell context predict the effects of the same compound in another?</p>
    <div class="callout"><strong>Main finding.</strong> The neural ensemble achieved {neural:.4f} mean profile RMSE, a {reduction:.1f}% reduction versus predicting no change. Its advantage over a linear model with matched target compression was only {ci['mean_rmse_improvement']:.4f}, with a 95% drug-cluster interval of [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]. These results do not establish a clear nonlinear advantage.</div>
    <h2>Data and experimental design</h2>
    <p>Real, public OP3 drug-perturbation measurements in human peripheral blood mononuclear cells. The supplied processed matrices contain 5,317 genes and four annotated cell types. Training used 306 conditions, including eight positive-control compound profiles; 96 public-validation conditions selected hyperparameters. Selected models were then refit on all 402 development conditions and evaluated on 151 official private-test conditions from 76 compounds in B and myeloid cells.</p>
    <p>Complete drug–cell-type pairs are held out. Each test compound and cell-type label is present in development data. The outcome is estimated log2 fold change relative to DMSO; this is a study of effect-size prediction, not the competition's transformed-p-value leaderboard task. All code-fitted vocabularies and target PCA use training outcomes only.</p>
    <h2>Five model recipes, one final test</h2>{table.to_html(float_format=lambda x: f'{x:.4f}', na_rep='—', border=0)}
    <p>The neural model is a compact one-hot drug/context network with a linear skip and nonlinear hidden layer, predicting 24 training-derived response components. Three fixed seeds are averaged. PCA ridge is a linear comparator with the same component count; it also performs strongly. Unregularized transfer of the same drug's mean response across training cell types is worse than predicting no change, showing that transfer requires care.</p>
    {figures[0]}{figures[1]}
    <h2>Reliability and biological interpretation</h2>
    <p>The neural model recovers {100*summary.loc['PyTorch ensemble','top50_overlap']:.1f}% of each condition's 50 largest observed absolute-effect genes, on average. This describes effect-size ranking, not statistically significant gene recovery. Ensemble disagreement had effectively no rank association with prediction error here (Spearman correlation: {rho:.2f}); it should not be used as a reliable confidence score.</p>
    {figures[2]}
    <p>Reactome release 93 provides 1,042 gene sets with at least ten measured genes, covering 3,455 distinct measured genes. Pathway scores average measured gene log fold changes; they are descriptive summaries and can cancel opposing changes. Pathways overlap and coverage is incomplete. The displayed condition, {html.escape(example_label)}, was selected by proximity to the median observed response norm.</p>
    {figures[4]}
    <p>In this example, both compressed models capture the negative interferon-related gene-set means, while missing several positive calcium-associated means. These are descriptions of measured expression patterns, not evidence that specific signaling pathways were activated or inhibited.</p>
    <h2>Choosing the next experiment</h2>
    <p>A secondary offline simulation reveals measured responses from a development-only acquisition pool. Both strategies start with the same 144 labeled conditions and evaluate on the same 96 public-validation conditions. At 264 labels, design-based ridge leverage averaged {final_al.loc['ridge_leverage','mean']:.4f} RMSE versus {final_al.loc['random','mean']:.4f} for random acquisition over five initializations.</p>
    <p>This demonstrates a reproducible experiment-selection loop under these conditions. Leverage is a design-based variance proxy, independent of hidden pool outcomes, and is not calibrated biological uncertainty. Public validation is reused for this descriptive study; the final private test is excluded throughout acquisition.</p>
    {figures[3]}
    <h2>What the study establishes</h2><ul>
    <li>Regularized response transfer improves on no-change and naive cross-context averages on these processed benchmark outcomes.</li>
    <li>Low-dimensional linear prediction captures much of the observed gain. Architecture, optimization, regularization and ensembling remain different across recipes.</li>
    <li>The experiment-selection simulation favors leverage at the examined budgets; generalization to other screens remains an open question.</li></ul>
    <h2>Limits and next experiments</h2><p>The original benchmark performs upstream gene filtering and parts of normalization/differential-expression estimation jointly across conditions, including test samples. This is evaluation on benchmark-provided processed outcomes, not fully isolated raw-data preprocessing. Responses aggregate donors, so donor generalization is untested. Compounds and context labels are known; new-drug or entirely new-cell-type performance is not assessed. There is no tissue geometry, explicit cell–cell signaling model, mechanistic inference or wet-lab validation.</p>
    <p>The next stronger tests are donor-resolved, raw-data-isolated preprocessing; an external perturbation dataset; better calibrated uncertainty; and prospective experimental validation. No state-of-the-art or therapeutic-efficacy claim is made.</p>
    <h2>Reproduce and inspect</h2><p><code>python scripts/download_data.py</code><br><code>python scripts/run_study.py</code><br><code>python scripts/make_report.py</code><br><code>python -m pytest -q</code></p>
    <p>Full source, pinned input checksums, hyperparameter selections, per-condition scores, paired intervals, acquisition histories, pathway coverage, environment versions and the local study plan accompany this report. Prepared with AI assistance; results come from executed code on the cited public data. This is an exploratory comparison of existing modeling approaches.</p>
    <p class="small">Sources: {sources}. See DATACARD.md and REFERENCES.md in the repository for processing and license details.</p></main></html>'''
    (ROOT / "REPORT.html").write_text(body)
    print(f"Report generated; {len(pathway_info)} Reactome summaries; uncertainty rho={rho:.3f}")


if __name__ == "__main__":
    main()
