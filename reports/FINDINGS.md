# Drug-response transfer across immune cell types

**Regularized models improved held-out response prediction, but this experiment did not establish an advantage for the nonlinear model over a matched linear comparator.** The PyTorch ensemble reduced mean profile RMSE by **10.4%** relative to predicting no change. Almost the same accuracy was obtained by ridge regression with training-only target PCA.

## Question and design

Can responses measured for a compound in some immune cell types predict its response in another? This study uses the processed [Open Problems perturbation-prediction benchmark](https://github.com/openproblems-bio/task_perturbation_prediction), with **553 drug–cell-type profiles and 5,317 supplied genes**. Outcomes are benchmark-provided log2 fold changes relative to DMSO.

The official split supplied 306 initial training profiles, including measured positive-control compounds, and 96 public-validation profiles. Hyperparameters were selected on validation RMSE. Selected models were then refitted on all 402 development profiles and evaluated on 151 final-test profiles covering 76 drugs. Test combinations were absent from training; their drug and cell-type identities were represented. The final test concerns B and myeloid cells.

PCA, categorical encodings, and model parameters were fitted on training responses only. The neural prediction averages three seeded fits. The primary metric averages gene-wise RMSE over condition profiles, giving each profile equal weight.

## Final-test results

| Model | Mean profile RMSE ↓ | Mean Pearson correlation ↑ | Top-50 effect-gene overlap ↑ |
|---|---:|---:|---:|
| No change | 0.35919 | Undefined | Undefined |
| Transferred drug mean | 0.39261 | 0.247 | 0.183 |
| Additive ridge | 0.33475 | 0.364 | 0.230 |
| PCA ridge | 0.32319 | 0.391 | 0.259 |
| PyTorch ensemble | 0.32188 | 0.404 | 0.280 |

Transferring a drug's average measured response performed worse than predicting no change: RMSE increased by 0.03342. Its positive correlations therefore should not be mistaken for good effect-size prediction.

The neural ensemble improved RMSE by 0.01287 over full-output additive ridge (95% drug-cluster bootstrap interval **0.00372 to 0.02326**). Against PCA ridge, however, its improvement was only **0.00131**, with an interval of **−0.00363 to 0.00689**. The matched compression control substantially changes the interpretation: these results do not support attributing the gain over ordinary ridge to nonlinearity. Optimization, regularization, and ensembling also differ between recipes.

Intervals use 2,000 paired drug-cluster bootstrap samples and describe variability across these drugs, conditional on the fitted models and dataset. Top-50 overlap is descriptive recovery of large absolute effects, not a differential-expression significance test.

## Secondary experiment-selection simulation

Five paired simulations compared random acquisition with uncalibrated, design-based ridge leverage. Both strategies started with 144 labeled profiles, acquired four batches of 30 from the remaining initial-training pool, and used the 96 public-validation profiles solely for evaluation during acquisition.

At 264 labeled profiles, mean RMSE was **0.36338** for leverage selection and **0.36603** for random selection. This small advantage is descriptive: evaluation reused public validation, all runs shared that evaluation set, and the variance proxy depends on the categorical design. It does not establish a general benefit from active learning or calibrated biological uncertainty.

## Scope and reproducibility

Checksummed inputs, the study plan, selected configurations, environment versions, per-profile scores, and acquisition histories accompany the code. Exact numerical reproduction can vary with library versions and numerical backends.

The supplied outcomes aggregate donors, and upstream joint preprocessing used information beyond the training split. This is a processed-benchmark study, not a fully inductive raw-data evaluation. It does not test new compounds, unseen cell types, donor generalization, mechanistic cell–cell signaling, tissue geometry, causal mechanisms, or therapeutic efficacy. No state-of-the-art claim is made.
