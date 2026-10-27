# Data card

This study uses the official processed **Open Problems Perturbation Prediction (OP3)** benchmark. The experiment measured transcriptional responses to small molecules in human peripheral blood mononuclear cells after 24 hours. Cellarity and the OP3 authors generated the experimental data; this repository contributes a modeling and evaluation study. See [REFERENCES.md](REFERENCES.md).

## Files and outcomes

The downloader retrieves these two files from the [official Open Problems S3 collection](https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/workflow-resources/neurips-2023-data/):

| File | Shape | SHA-256 |
|---|---|---|
| `de_train.h5ad` | 402 conditions × 5,317 genes | `532d0ffa616233ba571d2300813cffc8bcdd0f83a78de7119cb229df8334e375` |
| `de_test.h5ad` | 151 conditions × 5,317 genes | `5746d870d61e453b8b404b512dbe310b961f38f1a5b5ca1c67ca243d673e868d` |

Each row represents one drug–cell-type combination, not an individual cell or donor. The H5AD files have no `X` matrix. This study predicts `layers/logFC`: benchmark-provided log2 fold changes relative to DMSO. It does not train on the benchmark's signed p-value target. Gene symbols are stored in `var/_index`; all 5,317 supplied genes are retained. The inspected logFC matrices contain no nonfinite values.

## Split contract

| Role in this study | Official labels | B cells | Myeloid cells | NK cells | T cells | Total |
|---|---|---:|---:|---:|---:|---:|
| Initial fitting | `train` + `control` | 14 | 13 | 139 | 140 | 306 |
| Hyperparameter selection | `public_test` | 49 | 47 | 0 | 0 | 96 |
| Final evaluation | `private_test` | 76 | 75 | 0 | 0 | 151 |

The eight `control` rows are measured responses to **Belinostat and Dabrafenib**, each in four cell types. They are positive-control compounds, not zero-response DMSO profiles. Their measured outcomes are retained for fitting. The 402 development profiles contain 140 compounds, including these two controls. Rows are unique drug–cell-type combinations.

Hyperparameters are chosen on the 96 validation conditions. Selected models are then refit on all 402 development conditions and evaluated on the 151 final conditions. Final-test outcomes are not used for model selection. The final compounds and cell types occur in development data; their particular combinations are unmeasured there. This tests transfer to **missing combinations**, not prediction for wholly unseen compounds, unseen cell types, or new donors. The evaluation set covers B and myeloid cells only.

## Upstream processing and limits

The underlying experiment includes three healthy donors. The provided differential-expression outcomes aggregate their measurements, so donor generalization and donor-level uncertainty cannot be evaluated here.

OP3 supplied the processed outcomes. Its published workflow filters genes before splitting, using expression across the dataset. It estimates training differential expression with training, control, and public-test samples; final differential expression uses all splits. The workflow includes shared normalization and statistical modeling. Our fitting and tuning respect the released splits, but these are **benchmark-provided processed outcomes**, not an independently processed, prospective test cohort. Reprocessing raw counts with isolated training-only decisions would be a separate study.

These measurements describe gene-expression responses in mixed immune-cell cultures. They do not identify causal cell–cell communication, tissue geometry, clinical benefit, or therapeutic safety. Gene and pathway summaries are descriptive checks on predicted response patterns. Rankings by absolute fold change do not establish statistical significance.

## Data rights and attribution

The [official competition rules](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/rules) identify the competition data license as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute Cellarity and the OP3 authors, cite their paper, link the source and license, and identify transformations when reusing data. This repository downloads the official processed files and does not redistribute those source matrices. The new project code is MIT-licensed separately.

Pathway membership comes from **Reactome release 93**, using the [pinned GMT archive](https://download.reactome.org/93/ReactomePathways.gmt.zip). Reactome annotation data are [CC0](https://reactome.org/license). This is a fixed reference release, not a claim to use the latest release. Gene-set overlap is restricted to genes available in this benchmark.
