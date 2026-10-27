# References and provenance

## Experimental data and benchmark

Szalata, A., Benz, A., Cannoodt, R., et al. (2024). **A benchmark for prediction of transcriptomic responses to chemical perturbations across cell types.** *Advances in Neural Information Processing Systems*, 37, 20566–20616.

- [Official proceedings page](https://papers.nips.cc/paper_files/paper/2024/hash/24c4d51f3ef48dd2dbab78243ecb26a1-Abstract-Datasets_and_Benchmarks_Track.html)
- [Paper and supplementary material](https://papers.nips.cc/paper_files/paper/2024/file/24c4d51f3ef48dd2dbab78243ecb26a1-Paper-Datasets_and_Benchmarks_Track.pdf)
- [OP3 benchmark description and results](https://openproblems.bio/results/perturbation_prediction)
- [GEO accession GSE279945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945)
- [Official benchmark repository](https://github.com/openproblems-bio/task_perturbation_prediction)
- [Processed training outcomes](https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/workflow-resources/neurips-2023-data/de_train.h5ad)
- [Processed final-test outcomes](https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/workflow-resources/neurips-2023-data/de_test.h5ad)
- [Official competition data-license terms](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/rules): CC BY 4.0 for competition data. This is separate from the new code's MIT license.

Attribution: experimental data from Cellarity and the OP3 authors. This study uses released logFC outcomes, applies its own modeling and evaluation, and reports transformations in [DATACARD.md](DATACARD.md) and the study code. No new biological measurements were generated.

## Upstream processing inspected

These source files document the upstream filtering, split handling, and differential-expression calculations. They are cited for provenance; their code is not copied into this project.

- [Workflow](https://github.com/openproblems-bio/task_perturbation_prediction/blob/main/src/workflows/process_dataset/main.nf) — inspected Git blob SHA `b4093bb8cd4f6aa8e3b3a9bce3b08538f248df2a`.
- [Gene filtering](https://github.com/openproblems-bio/task_perturbation_prediction/blob/main/src/process_dataset/filter_vars/script.R) — inspected Git blob SHA `7566f1cfac94034991d5c7ed23aab7b5d549f364`.
- [Limma outcome estimation](https://github.com/openproblems-bio/task_perturbation_prediction/blob/main/src/process_dataset/run_limma/script.R) — inspected Git blob SHA `d30923ecadafb33a685efa023c82241e2e669134`.

The workflow URL above is a source reference; the recorded blob hashes identify file contents, not repository commits. The downloadable data are pinned independently by SHA-256 in [DATACARD.md](DATACARD.md).

## Pathway annotation

**Reactome Pathway Database**, release 93. Human pathway gene sets are used for descriptive comparisons of predicted and measured transcriptional responses.

- [Pinned release 93 gene-set archive](https://download.reactome.org/93/ReactomePathways.gmt.zip)
- [Official download index](https://reactome.org/download-data)
- [Reactome license](https://reactome.org/license): annotation data are CC0; this statement does not relicense Reactome diagrams, software, or branding.

Pathway membership is external biological annotation, not validation of pathway activation or a causal mechanism. Overlapping gene sets and incomplete coverage of the supplied gene universe affect interpretation.
