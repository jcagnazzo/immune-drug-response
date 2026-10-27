"""Descriptive Reactome gene-set response summaries, with pinned provenance.

A response is the arithmetic mean log fold-change of the measured genes in a
pathway. It is not enrichment, pathway activation, statistical significance,
or evidence of a causal mechanism. Opposite changes can cancel. Reactome sets
overlap and are hierarchical, so correlations across them are not independent
biological validations. No labels or response values select eligible pathways.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse


REACTOME_RELEASE = 93
REACTOME_URL = "https://download.reactome.org/93/ReactomePathways.gmt.zip"
REACTOME_SHA256 = "eb0ddcb257e5c51feee4cb61d0eb982e24f8f7e1d56362436ded30af9b040d88"


def download_reactome(directory):
    """Download pinned release 93 and record its checksum and source metadata.

    Data are CC0 per https://reactome.org/license. The official download index
    is https://reactome.org/download-data. Release 93 is deliberately fixed;
    replacing it with the moving current-release URL would change the study.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "ReactomePathways.r93.gmt.zip"
    if not target.exists():
        partial = target.with_suffix(".partial")
        with urllib.request.urlopen(REACTOME_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != REACTOME_SHA256:
            raise ValueError("Unexpected bytes for Reactome release 93.")
        partial.write_bytes(payload)
        partial.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != REACTOME_SHA256:
        raise ValueError("Cached Reactome gene sets failed checksum validation.")
    manifest = directory / "reactome_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "source": "Reactome Pathway Database", "release": REACTOME_RELEASE,
            "url": REACTOME_URL, "sha256": REACTOME_SHA256,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "license": "CC0", "license_url": "https://reactome.org/license",
            "symbols": "human gene symbols, exact matching without alias conversion",
        }, indent=2) + "\n")
    return target


def read_reactome(path):
    """Read official GMT rows (name, stable ID, gene symbols), human sets only."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            text = archive.read("ReactomePathways.gmt").decode("utf-8")
    else:
        text = path.read_text()
    pathways = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError(f"Invalid GMT row {line_number}.")
        name, stable_id, *symbols = fields
        if not stable_id.startswith("R-HSA-"):
            continue
        if stable_id in pathways:
            raise ValueError(f"Duplicate Reactome pathway: {stable_id}.")
        pathways[stable_id] = {"name": name, "genes": tuple(sorted(set(filter(None, symbols))))}
    if not pathways:
        raise ValueError("No human Reactome gene sets were found.")
    return pathways


def pathway_responses(y, genes, pathways, min_overlap=10):
    """Return (conditions x pathway mean-logFC frame, pathway coverage frame).

    Pathway eligibility uses only gene identifiers, never the effect sizes.
    Each measured gene receives equal weight within a pathway; unmeasured genes
    are excluded, not imputed. Coverage is reported to expose this limitation.
    """
    y = np.asarray(y, dtype=float)
    genes = list(genes)
    if y.ndim != 2 or y.shape[1] != len(genes) or len(set(genes)) != len(genes):
        raise ValueError("Response width must match unique gene symbols.")
    if not np.isfinite(y).all() or min_overlap < 1:
        raise ValueError("Responses must be finite and min_overlap >= 1.")
    lookup = {gene: i for i, gene in enumerate(genes)}
    row_indices, col_indices, weights, info = [], [], [], []
    for stable_id, entry in sorted(pathways.items()):
        members = set(entry["genes"])
        matched = sorted(lookup[gene] for gene in members if gene in lookup)
        if len(matched) < min_overlap:
            continue
        column = len(info)
        row_indices.extend(matched)
        col_indices.extend([column] * len(matched))
        weights.extend([1 / len(matched)] * len(matched))
        info.append({"pathway_id": stable_id, "name": entry["name"],
                     "overlap_genes": len(matched), "pathway_genes": len(members),
                     "coverage": len(matched) / len(members)})
    if not info:
        raise ValueError("No pathways meet the measured-gene overlap threshold.")
    info = pd.DataFrame(info).set_index("pathway_id")
    matrix = sparse.csc_matrix((weights, (row_indices, col_indices)), shape=(len(genes), len(info)))
    scores = (matrix.T @ y.T).T
    return pd.DataFrame(scores, columns=info.index), info


def evaluate_pathways(y_true, y_pred, genes, pathways, min_overlap=10):
    """Per-condition agreement across descriptive pathway mean-logFC summaries."""
    if np.shape(y_true) != np.shape(y_pred):
        raise ValueError("True and predicted response matrices must match.")
    truth, _ = pathway_responses(y_true, genes, pathways, min_overlap)
    pred, _ = pathway_responses(y_pred, genes, pathways, min_overlap)
    a, b = truth.to_numpy(), pred.to_numpy()
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)
    correlation = np.divide(
        np.sum(a_centered * b_centered, axis=1), denom,
        out=np.full(len(a), np.nan), where=denom > 1e-12,
    )
    return pd.DataFrame({
        "condition_index": np.arange(len(a)),
        "pathway_pearson": np.clip(correlation, -1, 1),
        "pathway_rmse": np.sqrt(np.mean((a - b) ** 2, axis=1)),
        "n_pathways": len(truth.columns),
    })


def top_pathway_responses(responses, pathway_info, n=10):
    """Rank each condition's largest absolute summaries for descriptive review."""
    if n < 1 or not responses.columns.equals(pathway_info.index):
        raise ValueError("Require n >= 1 and aligned pathway metadata.")
    rows = []
    for condition, values in enumerate(responses.to_numpy()):
        order = np.argsort(-np.abs(values), kind="stable")[:n]
        for rank, column in enumerate(order, 1):
            stable_id = responses.columns[column]
            rows.append({"condition_index": condition, "rank": rank,
                         "pathway_id": stable_id, "mean_logfc": float(values[column]),
                         **pathway_info.loc[stable_id].to_dict()})
    return pd.DataFrame(rows)
